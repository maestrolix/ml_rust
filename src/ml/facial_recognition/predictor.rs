use image::{DynamicImage, Rgba32FImage};
use ndarray::Array;
use ort::{inputs, GraphOptimizationLevel, Session};

use crate::ml::{facial_detection::post_processing::DetectionFaces, transforms::crop_face};

pub struct FaceRecognition {
    pub session: Session,
}

impl FaceRecognition {
    pub fn new(model_path: &str) -> Self {
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        FaceRecognition { session }
    }

    pub fn predict(&self, raw_image: &DynamicImage, faces: DetectionFaces) -> Vec<[f32; 512]> {
        let image = raw_image.to_rgba32f();

        faces
            .faces
            .iter()
            .map(|face| crop_face(&image, &face.landmarks, 112))
            .map(|crop| Self::get_tensor(&crop))
            .map(|tensor| self.calculate_embedding(tensor))
            .collect()
    }

    pub fn calculate_embedding(
        &self,
        image: ndarray::Array<f32, ndarray::Dim<[usize; 4]>>,
    ) -> [f32; 512] {
        let dim = image.dim();
        if dim.1 != 3 || dim.2 != 112 || dim.3 != 112 {
            panic!("Dimenstion should be [n, 3, 112, 112]");
        }

        let outputs = self.session.run(inputs![image].unwrap()).unwrap();

        let embedding = outputs[0].try_extract_tensor().unwrap();

        return embedding.as_slice().unwrap().try_into().unwrap();
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
