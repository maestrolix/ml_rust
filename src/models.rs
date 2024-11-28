use axum::body::Bytes;
use axum_typed_multipart::{FieldData, TryFromMultipart};
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct DetectedFaceOutput {
    pub score: f32,
    pub bbox: [f32; 4],
    pub landmarks: [(f32, f32); 5],
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct RecognizedFaceOutput {
    pub score: f32,
    pub bbox: [f32; 4],
    pub landmarks: [(f32, f32); 5],
    pub embedding: Vec<f32>,
}

impl RecognizedFaceOutput {
    pub fn from_mergers(face: &DetectedFaceOutput, embedding: Vec<f32>) -> Self {
        RecognizedFaceOutput {
            score: face.score,
            bbox: face.bbox,
            landmarks: face.landmarks,
            embedding,
        }
    }
}

#[derive(TryFromMultipart, Debug)]
pub struct ImageForm {
    #[form_data(limit = "unlimited")]
    pub image: FieldData<Bytes>,
}

#[derive(ToSchema, Debug)]
pub struct ImageFormUtopia {
    pub image: Vec<u8>,
}

#[derive(ToSchema, Debug, IntoParams, Deserialize)]
pub struct TextQuery {
    pub text: String,
}

#[derive(Debug, ToSchema, Deserialize, Serialize)]
pub struct Span {
    /// Начало интервала попадания лица (в секундах)
    pub start: f32,
    /// Конец интервала попадания лица (в секундах)
    pub end: f32,
    /// Максимальное совпадение лица в интервале
    pub max_score: f32,
}

#[derive(Debug, ToSchema, Deserialize, Serialize)]
pub struct VideoFacialRecognitionOutput {
    /// Интервалы попадания лица в видео
    pub spans: Vec<Span>,
}

#[derive(TryFromMultipart, Debug)]
pub struct VideoForm {
    #[form_data(limit = "unlimited")]
    pub search_face: FieldData<Bytes>,
    #[form_data(limit = "unlimited")]
    pub video: FieldData<Bytes>,
}

#[derive(ToSchema, Debug)]
pub struct VideoFormUtopia {
    /// Лицо искомого человека
    pub search_face: Vec<u8>,
    /// Видео в рамках которого осуществляется поиск
    pub video: Vec<u8>,
}
