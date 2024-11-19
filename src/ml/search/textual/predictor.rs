use itertools::Itertools;
use ndarray::Array;
use ort::{inputs, GraphOptimizationLevel, Session, SessionOutputs};
use tokenizers::{Encoding, Tokenizer};

pub struct ImageTextualize {
    session: Session,
    tokenizer: Tokenizer,
}

impl ImageTextualize {
    pub fn new(model_path: &str, text_model_for_tokenizer: &str) -> Self {
        let tokenizer = Self::create_tokenizer(text_model_for_tokenizer);
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        ImageTextualize { session, tokenizer }
    }

    pub fn predict(&self, text: &str) -> Vec<f32> {
        let preprocessed = self.tokenizer.encode(text, true).unwrap();

        let binding = vec![text.to_string()];
        let input_ids_vector = Self::get_input_ids_vector(preprocessed.clone(), &binding);

        let binding = vec![text.to_string()];
        let attention_mask_vector = Self::get_attention_mask_vector(preprocessed, &binding);

        let outputs = self
            .session
            .run(inputs![input_ids_vector, attention_mask_vector].unwrap())
            .unwrap();

        try_extract(outputs, 1)
    }

    fn create_tokenizer(text_model_for_tokenizer: &str) -> Tokenizer {
        let mut tokenizer = Tokenizer::from_pretrained(text_model_for_tokenizer, None).unwrap();
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));
        tokenizer
    }

    fn get_attention_mask_vector(
        preprocessed: Encoding,
        text: &Vec<String>,
    ) -> ndarray::Array<i64, ndarray::Dim<[usize; 2]>> {
        let attention_mask_vector: Vec<i64> = preprocessed
            .get_attention_mask()
            .iter()
            .map(|b| *b as i64)
            .collect::<Vec<i64>>();

        let mask_shape = (text.len(), attention_mask_vector.len() / text.len());

        Array::from_shape_vec(mask_shape, attention_mask_vector).unwrap()
    }

    fn get_input_ids_vector(
        preprocessed: Encoding,
        text: &Vec<String>,
    ) -> ndarray::Array<i64, ndarray::Dim<[usize; 2]>> {
        let input_ids_vector: Vec<i64> = preprocessed
            .get_ids()
            .iter()
            .map(|b| *b as i64)
            .collect::<Vec<i64>>();

        let ids_shape = (text.len(), input_ids_vector.len() / text.len());

        Array::from_shape_vec(ids_shape, input_ids_vector).unwrap()
    }
}

fn try_extract(outputs: SessionOutputs, embed_index: usize) -> Vec<f32> {
    let binding = outputs[embed_index].try_extract_tensor().unwrap();
    let embeddings = binding.view();
    let seq_len = embeddings
        .shape()
        .first()
        .ok_or("cannot find seq_len with index 0 in text embeddings")
        .unwrap();
    let embeddings: Vec<f32> = embeddings
        .iter()
        .copied()
        .chunks(*seq_len)
        .into_iter()
        .flatten()
        .collect();
    embeddings
}
