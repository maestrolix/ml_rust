use crate::ml::search::{ImageTextualize, ImageVisualize};
use crate::models::TextQuery;
use crate::utils::dyn_image_from_bytes;

pub async fn clip_textual(textualize: ImageTextualize, text_query: TextQuery) -> Vec<f32> {
    textualize.predict(&text_query.text)
}

pub async fn clip_visual(visualize: ImageVisualize, image_raw: &[u8]) -> Vec<f32> {
    let image = dyn_image_from_bytes(image_raw);

    visualize.predict(image)
}
