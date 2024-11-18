use image::{DynamicImage, GenericImageView};
use ort::SessionOutputs;

use crate::schemas::{Bbox, Face};

const EPS: f32 = 1.0e-7;

#[derive(Debug)]
pub struct DetectionFaces {
    // Координаты лица и черт лица устанавливаются относительно 640X640 (исправить)
    pub faces: Vec<Face>,
}

impl DetectionFaces {
    pub fn new(
        outputs: SessionOutputs,
        original_image: &DynamicImage,
        threshold: f32,
    ) -> DetectionFaces {
        let mut faces: Vec<Face> = vec![];

        let scores08 = &outputs[0].try_extract_tensor().unwrap();
        let scores16 = &outputs[1].try_extract_tensor().unwrap();
        let scores32 = &outputs[2].try_extract_tensor().unwrap();
        let bboxes08 = &outputs[3].try_extract_tensor().unwrap();
        let bboxes16 = &outputs[4].try_extract_tensor().unwrap();
        let bboxes32 = &outputs[5].try_extract_tensor().unwrap();
        let kpsses08 = &outputs[6].try_extract_tensor().unwrap();
        let kpsses16 = &outputs[7].try_extract_tensor().unwrap();
        let kpsses32 = &outputs[8].try_extract_tensor().unwrap();

        for index in 0..12800 {
            let score = scores08[[index, 0]];
            if score > threshold {
                let bbox = distance2bbox(index, 8, bboxes08);
                let landmarks = distance2kps(index, 8, kpsses08);

                faces.push(Face {
                    score,
                    bbox,
                    landmarks,
                });
            }
        }

        for index in 0..3200 {
            let score = scores16[[index, 0]];
            if score > threshold {
                let bbox = distance2bbox(index, 16, bboxes16);
                let landmarks = distance2kps(index, 16, kpsses16);

                faces.push(Face {
                    score,
                    bbox,
                    landmarks,
                });
            }
        }

        for index in 0..800 {
            let score = scores32[[index, 0]];
            if score > threshold {
                let bbox = distance2bbox(index, 32, bboxes32);
                let landmarks = distance2kps(index, 32, kpsses32);

                faces.push(Face {
                    score,
                    bbox,
                    landmarks,
                });
            }
        }

        faces.sort_by(|a, b| (a.score.partial_cmp(&b.score).unwrap()));

        let mut unique_faces = non_maximum_suppression(faces, 0.5);

        // Self::norm_coords(&mut unique_faces, original_image);

        DetectionFaces {
            faces: unique_faces,
        }
    }

    pub fn norm_coords(faces: &mut Vec<Face>, original_image: &DynamicImage) {
        let (orig_width, orig_height) = (
            original_image.width() as f32,
            original_image.height() as f32,
        );
        // Описать кейс с масштабом меньшим чем требуемый (пока делаю для большего чем требуется)
        let ratio = if orig_width / orig_height > 1. {
            orig_width / 640.
        } else if orig_width / orig_height < 1. {
            orig_height / 640.
        } else {
            1.
        };

        // let (x_compensation, y_compensation) = if orig_width / orig_height > 1. {
        //     (0., 640. - orig_height * ratio)
        // } else if orig_width / orig_height < 1. {
        //     (0., 0.)
        // } else {
        //     (0., 0.)
        // };

        for face in faces {
            // face.bbox = face.bbox.map(|el| el * ratio);
            face.landmarks = face.landmarks.map(|(x, y)| (x, y));
        }
    }
}

fn distance2bbox(
    index: usize,
    stride: usize,
    distance: &ndarray::prelude::ArrayBase<
        ndarray::ViewRepr<&f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
) -> Bbox {
    let m = 640 / stride;
    let x = ((index / 2) * stride) % 640;
    let y = (((index / 2) / m) * stride) % 640;

    let x1 = x as f32 - distance[[index, 0]] * stride as f32;
    let y1 = y as f32 - distance[[index, 1]] * stride as f32;

    let x2 = x as f32 + distance[[index, 2]] * stride as f32;
    let y2 = y as f32 + distance[[index, 3]] * stride as f32;

    return [x1, y1, x2, y2];
}

fn distance2kps(
    index: usize,
    stride: usize,
    distance: &ndarray::prelude::ArrayBase<
        ndarray::ViewRepr<&f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
) -> [(f32, f32); 5] {
    let m = 640 / stride;
    let x = ((index / 2) * stride) % 640;
    let y = (((index / 2) / m) * stride) % 640;

    let x1 = x as f32 + distance[[index, 0]] * stride as f32;
    let y1 = y as f32 + distance[[index, 1]] * stride as f32;

    let x2 = x as f32 + distance[[index, 2]] * stride as f32;
    let y2 = y as f32 + distance[[index, 3]] * stride as f32;

    let x3 = x as f32 + distance[[index, 4]] * stride as f32;
    let y3 = y as f32 + distance[[index, 5]] * stride as f32;

    let x4 = x as f32 + distance[[index, 6]] * stride as f32;
    let y4 = y as f32 + distance[[index, 7]] * stride as f32;

    let x5 = x as f32 + distance[[index, 8]] * stride as f32;
    let y5 = y as f32 + distance[[index, 9]] * stride as f32;

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)];
}

/// Run non-maximum-suppression on candidate bounding boxes.
///
/// The pairs of bounding boxes with confidences have to be sorted in **ascending** order of
/// confidence because we want to `pop()` the most confident elements from the back.
///
/// Start with the most confident bounding box and iterate over all other bounding boxes in the
/// order of decreasing confidence. Grow the vector of selected bounding boxes by adding only those
/// candidates which do not have a IoU scores above `max_iou` with already chosen bounding boxes.
/// This iterates over all bounding boxes in `sorted_bboxes_with_confidences`. Any candidates with
/// scores generally too low to be considered should be filtered out before.
fn non_maximum_suppression(
    mut sorted_bboxes_with_confidences: Vec<Face>,
    max_iou: f32,
) -> Vec<Face> {
    let mut selected: Vec<Face> = vec![];
    'candidates: loop {
        // Get next most confident bbox from the back of ascending-sorted vector.
        // All boxes fulfill the minimum confidence criterium.
        match sorted_bboxes_with_confidences.pop() {
            Some(face) => {
                // Check for overlap with any of the selected bboxes
                for selected_face in selected.iter() {
                    match iou(&face.bbox, &selected_face.bbox) {
                        x if x > max_iou => continue 'candidates,
                        _ => (),
                    }
                }

                // bbox has no large overlap with any of the selected ones, add it
                selected.push(face.clone())
            }
            None => break 'candidates,
        }
    }

    selected
}

/// Calculate the intersection-over-union metric for two bounding boxes.
fn iou(bbox_a: &Bbox, bbox_b: &Bbox) -> f32 {
    // Calculate corner points of overlap box
    // If the boxes do not overlap, the corner-points will be ill defined, i.e. the top left
    // corner point will be below and to the right of the bottom right corner point. In this case,
    // the area will be zero.
    let overlap_box: Bbox = [
        f32::max(bbox_a[0], bbox_b[0]),
        f32::max(bbox_a[1], bbox_b[1]),
        f32::min(bbox_a[2], bbox_b[2]),
        f32::min(bbox_a[3], bbox_b[3]),
    ];

    let overlap_area = bbox_area(&overlap_box);

    // Avoid division-by-zero with `EPS`
    overlap_area / (bbox_area(bbox_a) + bbox_area(bbox_b) - overlap_area + EPS)
}

/// Calculate the area enclosed by a bounding box.
///
/// The bounding box is passed as four-element array defining two points:
/// `[x_top_left, y_top_left, x_bottom_right, y_bottom_right]`
/// If the bounding box is ill-defined by having the bottom-right point above/to the left of the
/// top-left point, the area is zero.
fn bbox_area(bbox: &Bbox) -> f32 {
    let width = bbox[3] - bbox[1];
    let height = bbox[2] - bbox[0];
    if width < 0.0 || height < 0.0 {
        // bbox is empty/undefined since the bottom-right corner is above the top left corner
        return 0.0;
    }

    width * height
}
