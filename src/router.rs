use crate::ml::{
    facial_processing::{FaceDetector, FaceRecognizer},
    search::{ImageTextualize, ImageVisualize},
};
use crate::models::{
    DetectedFaceOutput, ImageForm, ImageFormUtopia, RecognizedFaceOutput, TextQuery,
};

use axum::{
    extract::{DefaultBodyLimit, Query},
    routing::post,
    Json, Router,
};
use axum_typed_multipart::TypedMultipart;
use image::ImageReader;
use image::{DynamicImage, EncodableLayout};
use std::io::Cursor;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub fn create_app() -> Router {
    #[derive(OpenApi)]
    #[openapi(
        paths(
            detecting_faces,
            recognition_faces,

            clip_textual,
            clip_visual,
        ),
        components(
            schemas(ImageFormUtopia, DetectedFaceOutput, RecognizedFaceOutput, TextQuery)
        ),
        tags(
            (name = "face-processing", description = "Работа с лицами"),
            (name = "search", description = "Поисковики"),
        )
    )]
    struct ApiDoc;

    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/detecting-faces", post(detecting_faces))
        .route("/recognition-faces", post(recognition_faces))
        .route("/clip-textual", post(clip_textual))
        .route("/clip-visual", post(clip_visual))
        .layer(DefaultBodyLimit::max(100000000))
}

#[utoipa::path(
    post,
    path = "/detecting-faces",
    tag = "face-processing",
    request_body(content_type="multipart/form-data", content=ImageFormUtopia),
    responses(
        (status = 200, description = "Информация обработана успешно", body = Vec<DetectedFaceOutput>)
    )
)]
pub async fn detecting_faces(
    TypedMultipart(image_form): TypedMultipart<ImageForm>,
) -> Json<Vec<DetectedFaceOutput>> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());

    let detector = FaceDetector::new(
        "/home/stepan/rust/projects/recognition_all/ml_rust/models/antelopev2/detection/model.onnx",
    );

    let faces = detector.predict(&image);

    Json(faces)
}

#[utoipa::path(
    post,
    path = "/recognition-faces",
    tag = "face-processing",
    request_body(content_type="multipart/form-data", content=ImageFormUtopia),
    responses(
        (status = 200, description = "Информация обработана успешно", body = Vec<RecognizedFaceOutput>)
    )
)]
pub async fn recognition_faces(
    TypedMultipart(image_form): TypedMultipart<ImageForm>,
) -> Json<Vec<RecognizedFaceOutput>> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());

    let detector = FaceDetector::new(
        "/home/stepan/rust/projects/recognition_all/ml_rust/models/antelopev2/detection/model.onnx",
    );
    let faces = detector.predict(&image);

    let recognizer = FaceRecognizer::new(
        "/home/stepan/rust/projects/recognition_all/ml_rust/models/antelopev2/recognition/model.onnx"
    );

    Json(
        faces
            .iter()
            .zip(recognizer.predict(&image, &faces))
            .map(|(face, emb)| RecognizedFaceOutput::from_mergers(face, emb.to_vec()))
            .collect(),
    )
}

#[utoipa::path(
    post,
    path = "/clip-textual",
    tag = "search",
    params(TextQuery),
    responses(
        (status = 200, description = "Информация обработана успешно", body = Vec<f32>)
    )
)]
pub async fn clip_textual(Query(text_query): Query<TextQuery>) -> Json<Vec<f32>> {
    let textualize = ImageTextualize::new(
        "/home/stepan/rust/projects/recognition_all/recognition/models/clip/text/model.onnx",
        "sentence-transformers/clip-ViT-B-32-multilingual-v1",
    );

    Json(textualize.predict(&text_query.text))
}

#[utoipa::path(
    post,
    path = "/clip-visual",
    tag = "search",
    request_body(content_type="multipart/form-data", content=ImageFormUtopia),
    responses(
        (content_type="multipart/form-data", status = 200, description = "Информация обработана успешно", body = Vec<f32>)
    )
)]
pub async fn clip_visual(TypedMultipart(image_form): TypedMultipart<ImageForm>) -> Json<Vec<f32>> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());

    let visualize = ImageVisualize::new(
        "/home/stepan/rust/projects/recognition_all/recognition/models/clip/image/model.onnx",
    );

    Json(visualize.predict(image))
}

fn dyn_image_from_bytes(image_bytes: &[u8]) -> DynamicImage {
    ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
}
