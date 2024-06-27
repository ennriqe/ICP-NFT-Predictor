use std::cell::RefCell;
use prost::Message;
use tract_onnx::prelude::*;
use ndarray::Array2;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

thread_local! {
    static MODEL: RefCell<Option<Model>> = RefCell::new(None);
}

const LINEAR_REGRESSION_MODEL: &'static [u8] = include_bytes!("../assets/linear_regression.onnx");

pub fn setup() -> TractResult<()> {
    let bytes = bytes::Bytes::from_static(LINEAR_REGRESSION_MODEL);
    let proto: tract_onnx::pb::ModelProto = tract_onnx::pb::ModelProto::decode(bytes)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 11)))? // Ensure the shape matches your model's expectation
        .into_optimized()?
        .into_runnable()?;
    MODEL.with_borrow_mut(|m| {
        *m = Some(model);
    });
    Ok(())
}

pub fn classify(data: Vec<f32>) -> Result<Vec<f32>, anyhow::Error> {
    if data.len() != 11 {
        return Err(anyhow::anyhow!("Input data must have 11 elements."));
    }

    MODEL.with_borrow(|model| {
        let model = model.as_ref().unwrap();
        let input = Array2::from_shape_vec((1, 11), data)?; // Create a 2D array with shape (1, 11)
        let result = model.run(tvec!(Tensor::from(input).into()))?; // Convert Tensor to TValue using .into()
        let output: Vec<f32> = result[0].to_array_view::<f32>()?.iter().copied().collect();
        Ok(output)
    })
}
