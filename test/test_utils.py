import re
from qwen_vl_utils import process_vision_info

# unicode conversion: char <-> int
# use chr() and ord()
# char_table = [chr(i) for i in range(32,127)]
# valid_voc = list(string.printable[:-6])
# invalid_voc=['□', '∫', 'æ', '⬏', 'Σ', '■', 'Å', 'Ḏ', '£', 'ń', '⌀', 'Ù', '│', 'Ⅶ', 'Â', 'ς', 'Ⅻ', '⁴', 'ъ', '∁', 'Æ', 'α', 'Ç', 'ˣ', '・', '⤤', 'Đ', 'ı', '≡', '⋄', 'Å', 'ᴴ', 'ᵗ', 'Ȃ', 'δ', 'Ì', 'Ρ', '⟷', 'ï', '«', 'ȯ', 'Ǒ', '⇩', 'ζ', '✰', '⁹', 'м', 'Ộ', '❘', '₄', '²', 'φ', '⌴', '⇨', 'ƌ', 'σ', 'Ⅸ', '∞', 'ţ', 'ů', '◁', '½', '¾', 'ᴾ', '�', 'ê', 'Ⅵ', 'ˢ', '°', 'ɮ', '⇪', 'ᵈ', 'Ė', 'Ǐ', '⊲', '·', 'û', '˅', '⊤', '↰', 'Ī', 'ȍ', '×', '⊝', '‟', '√', '➀', 'î', '↹', '➞', '↑', 'ü', '⋏', '℃', 'Û', 'Ȅ', '›', '⟶', '○', 'Ⓡ', 'Ȋ', '➜', 'ᴺ', 'å', '►', '˂', 'ι', 'ā', 'Ś', '∇', '•', '¥', '★', '⋅', 'ₖ', 'ũ', '⁼', 'İ', '∓', '⊂', '➯', '₅', 'Ồ', '»', 'Ž', 'ì', 'Ⅴ', '„', 'Ň', 'ú', '‑', 'Ä', '⊣', '˄', '˙', 'Ó', '±', '╳', 'ⁿ', 'ū', 'ş', 'л', 'Ṡ', 'ᴵ', 'Ȏ', 'ñ', 'λ', '✓', 'ø', '✞', '≤', 'Õ', '⎯', '⬌', 'ʳ', 'Š', '◉', '➨', 'ᶜ', 'ź', 'ġ', 'ÿ', '◦', 'ḻ', '➮', 'ᴸ', 'Ú', '─', '⇧', '⤶', 'ð', 'ë', 'Ξ', 'ȑ', '⇦', '↻', 'ă', 'Ě', 'Ω', 'Á', '₃', 'к', 'Ⅰ', '▬', '—', '∈', 'Ạ', '☐', '⁸', 'Ŕ', 'ù', 'â', 'п', 'ᴭ', '÷', '↲', '‘', 'Ȇ', 'ᵀ', '¿', 'Ț', '▎', 'ě', 'ⱽ', 'Λ', '∷', '△', 'ç', 'ǫ', 'Ầ', '➩', 'и', 'Ū', 'ý', '―', '⇵', 'Í', 'ꝋ', '↓', '©', '³', 'Ɔ', 'è', '🠈', 'ğ', 'Ⓐ', 'я', 'Φ', 'Ấ', 'ᵖ', '︽', '˚', 'œ', '∥', 'β', 'й', 'Ⓒ', '⬍', '∨', '℮', '¼', 'ć', '␣', 'Ã', '🡨', 'Ą', 'ǵ', '™', 'Ế', 'ᵐ', '◄', 'Ń', '✱', 'ô', '¢', '₁', 'Ⅱ', '¹', 'π', 'µ', 'Ĺ', '⍙', 'р', 'Ï', 'ε', '⟵', '∆', 'ы', '⧫', 'ã', 'ė', '⁰', '⬉', '−', '⬋', '◯', 'о', 'À', 'ρ', '☰', 'τ', 'ŗ', '⸬', 'Ö', 'é', 'ə', 'Ǫ', 'Ē', '⎵', '𝔀', 'ⓒ', 'ȏ', '“', 'Č', 'č', 'Î', '∙', 'ṣ', '\u200b', '✚', 'ō', '”', 'ö', 'ᴹ', '▢', 'ν', '⌣', '：', '︾', '﹘', 'а', '∖', '⌄', 'в', '︿', 'ᵃ', 'ớ', '↺', '▲', '▽', '…', 'Ë', '⌫', '⤷', '€', '⊘', 'Ŏ', '₂', '⤺', '⁵', 'Ȧ', '∧', 'ω', '卐', 'Ⅳ', '⁻', '↵', 'ĩ', 'Ⅲ', 'Ă', '⬸', 'ʃ', 'ȇ', '←', '⅓', '⮌', '⇥', 'η', '➦', 'Ô', '⬊', '℉', '⊥', 'á', 'ŉ', '⊚', '–', 'Ā', '∅', 'Ć', '∎', '⤸', '⦁', 'ē', 'ί', 'õ', 'ᴱ', 'υ', 'ß', '◡', 'È', '∣', 'Δ', 'ᴙ', 'ò', '⊢', 'κ', '☓', 'Ề', 'Θ', 'ä', '﹀', '☆', 'Ò', '˃', 'à', 'Ê', 'ʰ', 'Ğ', '’', '→', '®', '●', '⁺', 'Ţ', 'Ż', '̓', '▼', 'Ể', 'ᵒ', 'Ý', 'б', '➔', 'г', '∴', '⅔', '⬈', 'Ō', '∊', 'Π', 'Ⅷ', 'Ñ', '➝', 'É', 'Ł', 'ó', '∉', 'Ø', 'Ü', '⋮', 'ĺ', '≣', '∼', '↱', 'í', 'Ⅹ', 'ę', '⋯', 'с', '╎', '⤦', '⊼', 'ȧ', '∝', '⤻', 'ξ', 'š', '▾', 'γ', '¡', '⊳', 'д', '⁷', 'ж', '➧', 'ᴰ', '‧', '∘', 'ž', 'Ȯ', 'Ⅺ']
CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']


def decode(idxs):
    s = ''
    for idx in idxs:
        if idx < len(CTLABELS):
            s += CTLABELS[idx] 
        else:
            return s
    return s


def encode(word):
    s = []
    max_word_len = 25
    for i in range(max_word_len):
        if i < len(word):
            char=word[i]
            idx = CTLABELS.index(char)
            s.append(idx)
        else:
            s.append(96)
    return s


def remove_focus_sentences(text):
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense', 'locate', 'position']
    parts = re.split(r'([.?!])', text)
    
    filtered_sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''

        full_sentence = sentence + punctuation
        
        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break
        
        if not skip:
            filtered_sentences.append(full_sentence)
        
        i += 2
    
    return "".join(filtered_sentences).strip()


def vlm_initial_text_extraction(image_path, model, processor):
    
    
    question = "OCR this image and transcribe only the English text."
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {"type": "text", "text": f"{question}"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    clean_text = output_text[0].replace('\n', "")
    
    return clean_text