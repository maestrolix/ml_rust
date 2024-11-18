pub mod ml;
pub mod schemas;
use crate::ml::{
    facial_detection::predictor::FaceDetector, facial_recognition::predictor::FaceRecognition,
};
use ml::transforms::resize;
use rayon::prelude::*;

fn main() {
    let emb1 = get_emb("/home/stepan/rust/test_data/images_with_faces/10.jpg", "1");
    let emb2 = get_emb("/home/stepan/rust/test_data/images_with_faces/11.jpg", "2");

    let res = cosine_similarity(&emb1, &emb2, false);
    dbg!(res);
}

fn get_emb(path_from: &str, with: &str) -> [f32; 512] {
    let detector = FaceDetector::new(
        "/home/stepan/rust/projects/recognition_all/ml_rust/models/antelopev2/detection/model.onnx",
    );

    let image = image::open(path_from).unwrap();

    let faces = detector.predict(&image);

    dbg!(&faces);

    let recognition = FaceRecognition::new(
        "/home/stepan/rust/projects/recognition_all/ml_rust/models/antelopev2/recognition/model.onnx",
    );

    // let res = resize(&image, 640, 640);
    let res = image.resize_to_fill(640, 640, image::imageops::FilterType::Triangle);
    res.to_rgb8().save(format!("{with}.jpg")).unwrap();

    recognition.predict(&res, faces)[0]
}

pub fn cosine_similarity(vec1: &[f32; 512], vec2: &[f32; 512], normalized: bool) -> f32 {
    let dot_product: f32 = vec1
        .par_iter()
        .zip(vec2.par_iter())
        .map(|(a, b)| a * b)
        .sum();

    if normalized {
        dot_product
    } else {
        let magnitude1: f32 = vec1.par_iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let magnitude2: f32 = vec2.par_iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        dot_product / (magnitude1 * magnitude2)
    }
}
