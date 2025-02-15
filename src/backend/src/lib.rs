use std::cell::RefCell;
use candid::{CandidType, Deserialize};
use ic_stable_structures::{memory_manager::{MemoryId, MemoryManager}, DefaultMemoryImpl};

mod onnx;

const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

#[derive(CandidType, Deserialize)]
struct Classification {
    label: String,
    score: f32,
}

#[derive(CandidType, Deserialize)]
struct ClassificationError {
    message: String,
}

#[derive(CandidType, Deserialize)]
enum ClassificationResult {
    Ok(Vec<Classification>),
    Err(ClassificationError),
}

#[ic_cdk::update]
fn classify(input: Vec<f32>) -> ClassificationResult {
    if input.len() != 11 {
        return ClassificationResult::Err(ClassificationError {
            message: "Input must be a vector of 11 floats.".to_string(),
        });
    }

    let result = match onnx::classify(input) {
        Ok(scores) => {
            let classifications: Vec<Classification> = scores.into_iter().enumerate().map(|(i, score)| Classification {
                label: format!("Prediction_{}", i),
                score,
            }).collect();
            ClassificationResult::Ok(classifications)
        }
        Err(err) => ClassificationResult::Err(ClassificationError {
            message: err.to_string(),
        }),
    };
    result
}

#[ic_cdk::init]
fn init() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    onnx::setup().unwrap();
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    onnx::setup().unwrap();
}
