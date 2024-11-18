pub type Bbox = [f32; 4];

#[derive(Debug, Clone, Copy)]
pub struct Face {
    pub score: f32,
    pub bbox: Bbox,
    pub landmarks: [(f32, f32); 5],
}
