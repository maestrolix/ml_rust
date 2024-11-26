use crate::utils::dyn_image_from_bytes;

use image::EncodableLayout;

use crate::ml::facial_processing::{FaceDetector, FaceRecognizer};
use crate::models::{DetectedFaceOutput, ImageForm, RecognizedFaceOutput};

pub async fn detecting_faces(
    detector: FaceDetector,
    image_form: ImageForm,
) -> Vec<DetectedFaceOutput> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());
    let faces = detector.predict(&image);

    faces
}

pub async fn recognition_faces(
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    image_form: ImageForm,
) -> Vec<RecognizedFaceOutput> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());

    let faces = detector.predict(&image);

    faces
        .iter()
        .zip(recognizer.predict(&image, &faces))
        .map(|(face, emb)| RecognizedFaceOutput::from_mergers(face, emb.to_vec()))
        .collect()
}
