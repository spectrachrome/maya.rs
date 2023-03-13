use rust_bert::pipelines::ner::NERModel;

fn main() {
    let ner_model = NERModel::new(Default::default()).unwrap();

    let input = [
        "My name is Amy. I live in Paris.",
        "Paris is a city in France.",
    ];

    let output = ner_model.predict(&input);
}