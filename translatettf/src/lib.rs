use harfbuzz_wasm::{debug, Font, Glyph, GlyphBuffer};
use wasm_bindgen::prelude::*;
use std::collections::HashMap;

use t5_quantized::{ModelConditionalGeneration, ConditionalGenerationParams};

static WEIGHTS: &[u8; 64350016] = include_bytes!("../model.gguf");
static TOKENIZER: &[u8; 1389353] = include_bytes!("../tokenizer.json");
static CONFIG: &[u8; 1206] = include_bytes!("../config.json");

static mut MODEL: Option<ModelConditionalGeneration> = None;

static mut GENERATE_CACHE: Option<HashMap<String, String>> = None;

fn init_model() -> Result<ModelConditionalGeneration, String> {
    let model = ModelConditionalGeneration::load(WEIGHTS.to_vec(), TOKENIZER.to_vec(), CONFIG.to_vec());
    model
}

fn build_gen_params(prompt: &str) -> ConditionalGenerationParams {
    ConditionalGenerationParams {
        prompt: prompt.to_string(),
        temperature: 0.0,
        seed: 0,
        top_p: 1.0,
        repeat_penalty: 1.1,
        repeat_last_n: 1,
        max_length: Some(512),
    }
}

enum Value {
    Int(isize),
    Float(f64),
}

fn is_number(s: &str) -> bool {
    if let Ok(i) = s.parse() {  // inferred as isize from next line
        Some(Value::Int(i)); true
    } else if let Ok(f) = s.parse() {
        Some(Value::Float(f)); true
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn shape(
    _shape_plan: u32,
    font_ref: u32,
    buf_ref: u32,
    _features: u32,
    _num_features: u32,
) -> i32 {
    let font = Font::from_ref(font_ref);
    let mut buffer = GlyphBuffer::from_ref(buf_ref);

    // Get buffer as string
    let buf_u8: Vec<u8> = buffer.glyphs.iter().map(|g| g.codepoint as u8).collect();
    let str_buf = String::from_utf8_lossy(&buf_u8);
    let buf_len = str_buf.chars().count();

    debug(&format!("Buffer: {}", str_buf));

    // If the string is just a number, map characters directly
    if is_number(&str_buf) {
        buffer.glyphs = vec![];
        buffer.glyphs.extend(
            str_buf
            .chars()
            .enumerate()
            .map(|(idx, x)| Glyph {
                codepoint: x as u32,
                flags: 0,
                x_advance: 0,
                y_advance: 0,
                cluster: idx as u32,
                x_offset: 0,
                y_offset: 0,
            })
        );
    } else {
        // Ensure model is initialized
        unsafe {
            if MODEL.is_none() {
                debug("Initializing model");
                MODEL = match init_model() {
                    Ok(model) => Some(model),
                    Err(e) => {
                        debug(&format!("Error loading model: {}", e));
                        return 0;
                    }
                };
                debug("Model initialized");
            }
            if GENERATE_CACHE.is_none() {
                GENERATE_CACHE = Some(HashMap::new());
            }
        }

        // Get model
        let model: &mut ModelConditionalGeneration = unsafe { MODEL.as_mut().unwrap() };
        let cache: &mut HashMap<String, String> = unsafe { GENERATE_CACHE.as_mut().unwrap() };

        let punctuation = vec!['.', '!', '?'];
        let mut sentences = str_buf.split_inclusive(|c| punctuation.contains(&c)).peekable();

        let mut glyphs = vec![];
        let mut cluster_idx = 0;

        while let Some(sentence) = sentences.next()  {
            if sentence.is_empty() {
                continue;
            }
            if sentence.chars().count() == 1 && punctuation.contains(&sentence.chars().next().unwrap()) {
                glyphs.push(Glyph {
                    codepoint: sentence.chars().next().unwrap() as u32,
                    flags: 0,
                    x_advance: 0,
                    y_advance: 0,
                    cluster: cluster_idx as u32,
                    x_offset: 0,
                    y_offset: 0,
                });
                cluster_idx += 1;
                continue;
            }
            let output_str = if !sentences.peek().is_none() || punctuation.contains(&sentence.chars().last().unwrap()) {
                if cache.contains_key(sentence) {
                    debug(&format!("Cache hit: {}", sentence));
                    cache.get(sentence).unwrap().to_string()
                } else {
                    let prompt = format!("translate English to German:{}", sentence);
                    let gen_params = build_gen_params(&prompt);
                    let output = model.decode(gen_params);
                    match output {
                        Ok(output) => {
                            debug(&format!("Generation: {}", output.generation));
                            cache.insert(sentence.to_string(), output.generation.to_string());
                            output.generation
                        }
                        Err(e) => {
                            debug(&format!("Error decoding: {}", e));
                            sentence.to_string()
                        }
                    }
                }
            } else {
                sentence.to_string()
            };
            glyphs.extend(
                output_str
                .chars()
                .enumerate()
                .map(|(_, x)| Glyph {
                    codepoint: x as u32,
                    flags: 0,
                    x_advance: 0,
                    y_advance: 0,
                    cluster: if cluster_idx < buf_len { let cluster = cluster_idx; cluster_idx += 1; cluster } else { buf_len - 1 } as u32,
                    x_offset: 0,
                    y_offset: 0,
                })
            );
        }
        buffer.glyphs = glyphs;
    }

    for item in buffer.glyphs.iter_mut() {
        // Map character to glyph
        item.codepoint = font.get_glyph(item.codepoint, 0);
        // Set advance width
        item.x_advance = font.get_glyph_h_advance(item.codepoint);
    }
    // Buffer is written back to HB on drop
    1
}
