use rust_bert::pipelines::ner::NERModel;

fn main() {
    let ner_model = NERModel::new(Default::default()).unwrap();

    let input = [
        "Turn on the lights in the living room",
        "Turn off the fan in the bedroom",
        "Increase the temperature in the kitchen",
        "Set the thermostat to 22 degrees",
        "Turn on the microwave",
        "Lights off",
    ];

    let output = ner_model.predict(&input);

    println!("{:#?}", output);
}