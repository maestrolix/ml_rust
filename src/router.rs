use crate::ml::{
    facial_processing::{FaceDetector, FaceRecognizer},
    search::{ImageTextualize, ImageVisualize},
};
use crate::models::{
    DetectedFaceOutput, ImageForm, ImageFormUtopia, RecognizedFaceOutput, TextQuery,
};

use axum::{
    extract::{DefaultBodyLimit, FromRef, Query, State},
    routing::post,
    Json, Router,
};
use axum_typed_multipart::TypedMultipart;
use image::ImageReader;
use image::{DynamicImage, EncodableLayout};
use std::io::Cursor;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[derive(Clone)]
pub struct AppState {
    pub detecrot: FaceDetector,
    pub recognizer: FaceRecognizer,
    pub textual: ImageTextualize,
    pub visual: ImageVisualize,
}

impl AppState {
    pub fn new(config: crate::config::Config) -> Self {
        AppState {
            detecrot: FaceDetector::new(
                config.model.facial_processing.detector.model_path,
                config.model.facial_processing.detector.model_name,
            ),
            recognizer: FaceRecognizer::new(
                config.model.facial_processing.recognizer.model_path,
                config.model.facial_processing.recognizer.model_name,
            ),
            textual: ImageTextualize::new(
                config.model.search.textual.model_path,
                config.model.search.textual.model_name,
            ),
            visual: ImageVisualize::new(
                config.model.search.visual.model_path,
                config.model.search.visual.model_name,
            ),
        }
    }
}

impl FromRef<AppState> for FaceDetector {
    fn from_ref(app_state: &AppState) -> FaceDetector {
        app_state.detecrot.clone()
    }
}

impl FromRef<AppState> for FaceRecognizer {
    fn from_ref(app_state: &AppState) -> FaceRecognizer {
        app_state.recognizer.clone()
    }
}

impl FromRef<AppState> for ImageTextualize {
    fn from_ref(app_state: &AppState) -> ImageTextualize {
        app_state.textual.clone()
    }
}

impl FromRef<AppState> for ImageVisualize {
    fn from_ref(app_state: &AppState) -> ImageVisualize {
        app_state.visual.clone()
    }
}

pub fn create_app(swagger_path: String, body_limit: u32, config: crate::config::Config) -> Router {
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

    let state = AppState::new(config);

    Router::new()
        .merge(SwaggerUi::new(swagger_path).url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/detecting-faces", post(detecting_faces))
        .route("/recognition-faces", post(recognition_faces))
        .route("/clip-textual", post(clip_textual))
        .route("/clip-visual", post(clip_visual))
        .with_state(state)
        .layer(DefaultBodyLimit::max(body_limit as _))
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
    State(detector): State<FaceDetector>,
    TypedMultipart(image_form): TypedMultipart<ImageForm>,
) -> Json<Vec<DetectedFaceOutput>> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());
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
    State(detector): State<FaceDetector>,
    State(recognizer): State<FaceRecognizer>,
    TypedMultipart(image_form): TypedMultipart<ImageForm>,
) -> Json<Vec<RecognizedFaceOutput>> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());

    let faces = detector.predict(&image);

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
pub async fn clip_textual(
    State(textualize): State<ImageTextualize>,
    Query(text_query): Query<TextQuery>,
) -> Json<Vec<f32>> {
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
pub async fn clip_visual(
    State(visualize): State<ImageVisualize>,
    TypedMultipart(image_form): TypedMultipart<ImageForm>,
) -> Json<Vec<f32>> {
    let image = dyn_image_from_bytes(image_form.image.contents.as_bytes());

    Json(visualize.predict(image))
}

fn dyn_image_from_bytes(image_bytes: &[u8]) -> DynamicImage {
    ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
}
