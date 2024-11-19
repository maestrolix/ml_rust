use ml_rust::ml::facial_processing::umeyama;

const SRC: [(f32, f32); 5] = [
    (491.7426, 321.8467),
    (541.7967, 332.23264),
    (507.47015, 366.4012),
    (485.72678, 369.63214),
    (533.7206, 378.40567),
];

const DST: [(f32, f32); 5] = [
    (38.2946f32, 51.6963f32),
    (73.5318f32, 51.5014f32),
    (56.0252f32, 71.7366f32),
    (41.5493f32, 92.3655f32),
    (70.7299f32, 92.2041f32),
];

#[test]
fn estimate() {
    let result = umeyama(&SRC, &DST);

    const R: nalgebra::Matrix<
        f32,
        nalgebra::Const<3>,
        nalgebra::Const<3>,
        nalgebra::ArrayStorage<f32, 3, 3>,
    > = nalgebra::Matrix3::<f32>::new(
        0.7000762f32,
        0.12366913f32,
        -346.2191f32,
        -0.12366913f32,
        0.7000762f32,
        -112.38885f32,
        0f32,
        0f32,
        1f32,
    );

    assert_eq!(result, R);
}
