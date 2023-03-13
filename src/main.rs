use rust_bert::pipelines::ner::NerModel;

fn main() -> anyhow::Result<()> {
    // Define the entities
    let entities = vec!["LIGHTS", "ROOM"];

    // Collect training data using distant supervision
    let train_data = vec![
        ("Turn on the lights in the living room", "Turn on the [B-LIGHTS]lights in the [B-ROOM]living room[E-ROOM][E-LIGHTS]"),
        ("Dim the lights in the bedroom", "Dim the [B-LIGHTS]lights in the [B-ROOM]bedroom[E-ROOM][E-LIGHTS]"),
        ("Set the thermostat to 70 degrees in the living room", "Set the thermostat to 70 degrees in the [B-ROOM]living room[E-ROOM]"),
        ("Turn off the TV in the den", "Turn off the [B-APPLIANCE]TV in the [B-ROOM]den[E-ROOM][E-APPLIANCE]"),
    ];

    // Train the NER system
    let mut model = NERModel::new(Default::default())?;
    model.train(&train_data)?;

    // Evaluate the NER system
    let eval_data = vec![
        ("Turn on the lights in the kitchen", "Turn on the [B-LIGHTS]lights in the [B-ROOM]kitchen[E-ROOM][E-LIGHTS]"),
        ("Lock the front door", "Lock the [B-DOOR]front door[E-DOOR]"),
        ("Set the temperature to 72 degrees", "Set the temperature to 72 degrees"),
    ];

    for (input, expected_output) in eval_data {
        let output = model.predict(&[input.to_owned()])?;
        println!("Input: {}", input);
        println!("Expected output: {}", expected_output);
        println!("Actual output: {:?}", output[0]);
        println!();
    }

    Ok(())
}
