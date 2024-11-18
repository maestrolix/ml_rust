use image::{DynamicImage, GenericImageView, Rgba32FImage};
use ndarray::Array;
use ort::{inputs, GraphOptimizationLevel, Session};

use crate::ml::facial_detection::post_processing::DetectionFaces;

use crate::ml::transforms::resize;

pub struct FaceDetector {
    pub session: Session,
}

impl FaceDetector {
    pub fn new(model_path: &str) -> Self {
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        FaceDetector { session }
    }

    pub fn predict(&self, image: &DynamicImage) -> DetectionFaces {
        // Некорректное изменение размера изображения (обрезает)
        let resized_image = image.resize_to_fill(640, 640, image::imageops::FilterType::Triangle);

        // let resized_image = resize(image, 640, 640);

        dbg!(&resized_image.dimensions());

        let image_tensor = Self::get_tensor(&resized_image.to_rgba32f());

        let outputs = self.session.run(inputs![image_tensor].unwrap()).unwrap();

        DetectionFaces::new(outputs, image, 0.5)
    }

    pub fn get_tensor(image: &Rgba32FImage) -> ndarray::Array<f32, ndarray::Dim<[usize; 4]>> {
        let shape = image.dimensions();

        let input = Array::from_shape_fn(
            (1_usize, 3_usize, shape.0 as usize, shape.1 as usize),
            |(_, c, i, j)| ((image[(j as _, i as _)][c] as f32) - 0.5f32) / 0.5f32,
        );

        return input;
    }
}
