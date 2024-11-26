pub mod config;
pub mod ml;
pub mod models;
pub mod router;
pub mod services;
pub mod utils;

#[tokio::main]
async fn main() {
    env_logger::init();

    let config = config::Config::new("config.toml");

    let addr = format!("{}:{}", config.service.host, config.service.port);

    let app = router::create_app(
        config.service.swagger_path.clone(),
        config.service.body_limit,
        config.clone(),
    );

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
