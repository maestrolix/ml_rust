use image::{DynamicImage, ImageReader};
use std::io::Cursor;

pub fn dyn_image_from_bytes(image_bytes: &[u8]) -> DynamicImage {
    ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
}
