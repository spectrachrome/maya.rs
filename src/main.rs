use rust_bert::pipelines::ner::NERModel;
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{BertVocab, Vocab};

fn main() {
    // Load the pre-trained NERModel
    let config = Config::from_pretrained(
        Resource::Remote(RemoteResource::from_pretrained(NERModel::NAME)),
        NERModel::CONFIG_FILE,
    );
    let mut ner_model = NERModel::new(config).unwrap();

    // Define the corpus of text and the keywords or patterns
    let corpus = [
        "Turn on the lights in the living room",
        "Turn off the fan in the bedroom",
        "Increase the temperature in the kitchen",
        "Set the thermostat to 22 degrees",
        "Turn on the microwave",
        "Lights off",
    ];
    let keywords = vec![
        "turn on",
        "turn off",
        "increase",
        "decrease",
        "set",
        "temperature",
        "light",
        "fan",
        "microwave",
    ];

    // Automatically label the corpus of text using distant supervision
    let labeled_data = ner_model.distant_supervision(&corpus, &keywords);

    // Define the label mapping
    let label_mapping = vec![
        ("B-DEVICE", 1),
        ("B-ROOM", 2),
        ("B-ATTRIBUTE", 3),
        ("B-VALUE", 4),
    ]
    .into_iter()
    .map(|(label, id)| (label.to_string(), id))
    .collect::<Vec<_>>();

    // Define the tokenizer and tokenization parameters
    let vocab = BertVocab::from_pretrained(
        Resource::Remote(RemoteResource::from_pretrained(BertVocab::NAME)),
        BertVocab::FILES,
    );
    let tokenizer = BertTokenizer::from_vocab(vocab, true);
    let max_length = 64;
    let truncation_strategy = TruncationStrategy::LongestFirst;

    // Fine-tune the NERModel on the labeled data
    let train_results = ner_model.train(
        &labeled_data,
        &label_mapping,
        2,
        tokenizer,
        max_length,
        truncation_strategy,
    );
    println!("{:#?}", train_results);

    // Save the fine-tuned model for future use
    ner_model.save_pretrained("fine_tuned_ner_model_voice_assistant").unwrap();

    // Test the NERModel on new data
    let test_input = [
        "Turn on the oven in the kitchen",
        "Turn off the lights in the bedroom",
        "Increase the fan speed in the living room",
        "Set the temperature to 20 degrees",
        "Turn on the TV in the family room",
        "Close the curtains in the bedroom",
    ];
    let test_output = ner_model.predict(&test_input);
    println!("{:#?}", test_output);
}