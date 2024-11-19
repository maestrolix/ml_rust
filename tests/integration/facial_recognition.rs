use ml_rust::ml::facial_processing::{FaceDetector, FaceRecognizer};
use rayon::prelude::*;

const TEST_DATA_DIR: &str = "./tests/assets/with_faces";

fn get_face_embedding(
    detector: &FaceDetector,
    recognizer: &FaceRecognizer,
    image_name: &str,
) -> [f32; 512] {
    let image = image::open(format!("{TEST_DATA_DIR}/{image_name}")).unwrap();
    let faces = detector.predict(&image);
    let embeddings = recognizer.predict(&image, &faces);
    embeddings[0]
}

pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot_product: f32 = vec1
        .par_iter()
        .zip(vec2.par_iter())
        .map(|(a, b)| a * b)
        .sum();

    let magnitude1: f32 = vec1.par_iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let magnitude2: f32 = vec2.par_iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    dot_product / (magnitude1 * magnitude2)
}

#[test]
fn check_similarity() {
    let detector = FaceDetector::new(
        "/home/stepan/rust/projects/recognition_all/ml_rust/models/antelopev2/detection/model.onnx"
            .to_string(),
        "".to_string(),
    );
    let recognizer = FaceRecognizer::new(
        "/home/stepan/rust/projects/recognition_all/ml_rust/models/antelopev2/recognition/model.onnx".to_string(),
        "".to_string(),
    );
    let (face1, face2, face3) = (
        get_face_embedding(&detector, &recognizer, "test_face_6.jpg"),
        get_face_embedding(&detector, &recognizer, "test_face_7.jpg"),
        get_face_embedding(&detector, &recognizer, "test_face_8.jpg"),
    );

    assert!(cosine_similarity(&face1, &face2) >= 0.817);
    assert!(cosine_similarity(&face2, &face3) >= 0.81);
    assert!(cosine_similarity(&face1, &face3) >= 0.857);
}
