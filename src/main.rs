pub mod ml;
pub mod models;
pub mod router;

#[tokio::main]
async fn main() {
    env_logger::init();
    let app = router::create_app();
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3003").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
