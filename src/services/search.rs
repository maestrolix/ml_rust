use crate::utils::dyn_image_from_bytes;

use image::EncodableLayout;

use crate::ml::search::{ImageTextualize, ImageVisualize};
use crate::models::{ImageForm, TextQuery};

pub async fn clip_textual(textualize: ImageTextualize, text_query: TextQuery) -> Vec<f32> {
    textualize.predict(&text_query.text)
}

pub async fn clip_visual(visualize: ImageVisualize, image_form: ImageForm) -> Vec<f32> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());

    visualize.predict(image)
}
