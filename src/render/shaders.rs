//! Built-in SPIR-V modules.
//!
//! The renderer needs two fixed shaders and no run time shader selection. Their
//! instruction streams are assembled directly into SPIR-V words, which keeps
//! shader creation inside the Rust build and avoids a compiler executable in
//! the build or deployment environment.
//!
//! The fragment path always multiplies the sampled value by the vertex colour.
//! RGBA textures use the identity component mapping, while single channel font
//! textures map one into RGB and red into alpha.

const OP_CAPABILITY: u16 = 17;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_CONSTANT_COMPOSITE: u16 = 44;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;

const CAPABILITY_SHADER: u32 = 1;
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_GLSL450: u32 = 1;
const EXECUTION_VERTEX: u32 = 0;
const EXECUTION_FRAGMENT: u32 = 4;
const EXECUTION_ORIGIN_UPPER_LEFT: u32 = 7;

const STORAGE_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_INPUT: u32 = 1;
const STORAGE_OUTPUT: u32 = 3;
const STORAGE_PUSH_CONSTANT: u32 = 9;

const DECORATION_BLOCK: u32 = 2;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_LOCATION: u32 = 30;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const BUILT_IN_POSITION: u32 = 0;

struct Module {
    words: Vec<u32>,
}

impl Module {
    fn new(bound: u32) -> Module {
        Module {
            words: vec![0x0723_0203, 0x0001_0000, 0, bound, 0],
        }
    }

    fn instruction(&mut self, opcode: u16, operands: &[u32]) {
        let word_count = operands.len() as u32 + 1;
        self.words.push((word_count << 16) | opcode as u32);
        self.words.extend_from_slice(operands);
    }

    fn string_instruction(
        &mut self,
        opcode: u16,
        prefix: &[u32],
        text: &str,
        suffix: &[u32],
    ) {
        let mut operands = Vec::with_capacity(prefix.len() + suffix.len() + text.len() / 4 + 1);
        operands.extend_from_slice(prefix);
        append_string(&mut operands, text);
        operands.extend_from_slice(suffix);
        self.instruction(opcode, &operands);
    }

    fn finish(self) -> Vec<u32> {
        self.words
    }
}

fn append_string(words: &mut Vec<u32>, text: &str) {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
}

pub fn ui_vertex() -> Vec<u32> {
    const VOID: u32 = 1;
    const FUNCTION_TYPE: u32 = 2;
    const FLOAT: u32 = 3;
    const VEC2: u32 = 4;
    const VEC4: u32 = 5;
    const UINT: u32 = 6;
    const PTR_INPUT_VEC2: u32 = 7;
    const PTR_INPUT_VEC4: u32 = 8;
    const PTR_OUTPUT_VEC2: u32 = 9;
    const PTR_OUTPUT_VEC4: u32 = 10;
    const PER_VERTEX: u32 = 11;
    const PTR_OUTPUT_PER_VERTEX: u32 = 12;
    const PUSH: u32 = 13;
    const PTR_PUSH: u32 = 14;
    const PTR_PUSH_VEC2: u32 = 15;
    const IN_POS: u32 = 16;
    const IN_UV: u32 = 17;
    const IN_COLOR: u32 = 18;
    const OUT_UV: u32 = 19;
    const OUT_COLOR: u32 = 20;
    const OUT_PER_VERTEX: u32 = 21;
    const PUSH_VALUE: u32 = 22;
    const ZERO_UINT: u32 = 23;
    const TWO_FLOAT: u32 = 24;
    const TWO_VEC2: u32 = 25;
    const ONE_FLOAT: u32 = 26;
    const ONE_VEC2: u32 = 27;
    const ZERO_FLOAT: u32 = 28;
    const MAIN: u32 = 29;
    const LABEL: u32 = 30;
    const POSITION: u32 = 31;
    const VIEWPORT_PTR: u32 = 32;
    const VIEWPORT: u32 = 33;
    const NORMALIZED: u32 = 34;
    const SCALED: u32 = 35;
    const NDC: u32 = 36;
    const NDC_X: u32 = 37;
    const NDC_Y: u32 = 38;
    const CLIP_POSITION: u32 = 39;
    const POSITION_PTR: u32 = 40;
    const UV: u32 = 41;
    const COLOR: u32 = 42;

    let mut module = Module::new(43);
    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_VERTEX, MAIN],
        "main",
        &[IN_POS, IN_UV, IN_COLOR, OUT_UV, OUT_COLOR, OUT_PER_VERTEX],
    );

    module.instruction(OP_DECORATE, &[IN_POS, DECORATION_LOCATION, 0]);
    module.instruction(OP_DECORATE, &[IN_UV, DECORATION_LOCATION, 1]);
    module.instruction(OP_DECORATE, &[IN_COLOR, DECORATION_LOCATION, 2]);
    module.instruction(OP_DECORATE, &[OUT_UV, DECORATION_LOCATION, 0]);
    module.instruction(OP_DECORATE, &[OUT_COLOR, DECORATION_LOCATION, 1]);
    module.instruction(OP_DECORATE, &[PER_VERTEX, DECORATION_BLOCK]);
    module.instruction(
        OP_MEMBER_DECORATE,
        &[PER_VERTEX, 0, DECORATION_BUILT_IN, BUILT_IN_POSITION],
    );
    module.instruction(OP_DECORATE, &[PUSH, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[PUSH, 0, DECORATION_OFFSET, 0]);

    module.instruction(OP_TYPE_VOID, &[VOID]);
    module.instruction(OP_TYPE_FUNCTION, &[FUNCTION_TYPE, VOID]);
    module.instruction(OP_TYPE_FLOAT, &[FLOAT, 32]);
    module.instruction(OP_TYPE_VECTOR, &[VEC2, FLOAT, 2]);
    module.instruction(OP_TYPE_VECTOR, &[VEC4, FLOAT, 4]);
    module.instruction(OP_TYPE_INT, &[UINT, 32, 0]);
    module.instruction(OP_TYPE_STRUCT, &[PER_VERTEX, VEC4]);
    module.instruction(OP_TYPE_STRUCT, &[PUSH, VEC2]);
    module.instruction(OP_TYPE_POINTER, &[PTR_INPUT_VEC2, STORAGE_INPUT, VEC2]);
    module.instruction(OP_TYPE_POINTER, &[PTR_INPUT_VEC4, STORAGE_INPUT, VEC4]);
    module.instruction(OP_TYPE_POINTER, &[PTR_OUTPUT_VEC2, STORAGE_OUTPUT, VEC2]);
    module.instruction(OP_TYPE_POINTER, &[PTR_OUTPUT_VEC4, STORAGE_OUTPUT, VEC4]);
    module.instruction(
        OP_TYPE_POINTER,
        &[PTR_OUTPUT_PER_VERTEX, STORAGE_OUTPUT, PER_VERTEX],
    );
    module.instruction(OP_TYPE_POINTER, &[PTR_PUSH, STORAGE_PUSH_CONSTANT, PUSH]);
    module.instruction(OP_TYPE_POINTER, &[PTR_PUSH_VEC2, STORAGE_PUSH_CONSTANT, VEC2]);

    module.instruction(OP_VARIABLE, &[PTR_INPUT_VEC2, IN_POS, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[PTR_INPUT_VEC2, IN_UV, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[PTR_INPUT_VEC4, IN_COLOR, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[PTR_OUTPUT_VEC2, OUT_UV, STORAGE_OUTPUT]);
    module.instruction(OP_VARIABLE, &[PTR_OUTPUT_VEC4, OUT_COLOR, STORAGE_OUTPUT]);
    module.instruction(
        OP_VARIABLE,
        &[PTR_OUTPUT_PER_VERTEX, OUT_PER_VERTEX, STORAGE_OUTPUT],
    );
    module.instruction(OP_VARIABLE, &[PTR_PUSH, PUSH_VALUE, STORAGE_PUSH_CONSTANT]);

    module.instruction(OP_CONSTANT, &[UINT, ZERO_UINT, 0]);
    module.instruction(OP_CONSTANT, &[FLOAT, TWO_FLOAT, 2.0f32.to_bits()]);
    module.instruction(OP_CONSTANT_COMPOSITE, &[VEC2, TWO_VEC2, TWO_FLOAT, TWO_FLOAT]);
    module.instruction(OP_CONSTANT, &[FLOAT, ONE_FLOAT, 1.0f32.to_bits()]);
    module.instruction(OP_CONSTANT_COMPOSITE, &[VEC2, ONE_VEC2, ONE_FLOAT, ONE_FLOAT]);
    module.instruction(OP_CONSTANT, &[FLOAT, ZERO_FLOAT, 0.0f32.to_bits()]);

    module.instruction(OP_FUNCTION, &[VOID, MAIN, 0, FUNCTION_TYPE]);
    module.instruction(OP_LABEL, &[LABEL]);
    module.instruction(OP_LOAD, &[VEC2, POSITION, IN_POS]);
    module.instruction(
        OP_ACCESS_CHAIN,
        &[PTR_PUSH_VEC2, VIEWPORT_PTR, PUSH_VALUE, ZERO_UINT],
    );
    module.instruction(OP_LOAD, &[VEC2, VIEWPORT, VIEWPORT_PTR]);
    module.instruction(OP_FDIV, &[VEC2, NORMALIZED, POSITION, VIEWPORT]);
    module.instruction(OP_FMUL, &[VEC2, SCALED, NORMALIZED, TWO_VEC2]);
    module.instruction(OP_FSUB, &[VEC2, NDC, SCALED, ONE_VEC2]);
    module.instruction(OP_COMPOSITE_EXTRACT, &[FLOAT, NDC_X, NDC, 0]);
    module.instruction(OP_COMPOSITE_EXTRACT, &[FLOAT, NDC_Y, NDC, 1]);
    module.instruction(
        OP_COMPOSITE_CONSTRUCT,
        &[VEC4, CLIP_POSITION, NDC_X, NDC_Y, ZERO_FLOAT, ONE_FLOAT],
    );
    module.instruction(
        OP_ACCESS_CHAIN,
        &[PTR_OUTPUT_VEC4, POSITION_PTR, OUT_PER_VERTEX, ZERO_UINT],
    );
    module.instruction(OP_STORE, &[POSITION_PTR, CLIP_POSITION]);
    module.instruction(OP_LOAD, &[VEC2, UV, IN_UV]);
    module.instruction(OP_STORE, &[OUT_UV, UV]);
    module.instruction(OP_LOAD, &[VEC4, COLOR, IN_COLOR]);
    module.instruction(OP_STORE, &[OUT_COLOR, COLOR]);
    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

pub fn ui_fragment() -> Vec<u32> {
    const VOID: u32 = 1;
    const FUNCTION_TYPE: u32 = 2;
    const FLOAT: u32 = 3;
    const VEC2: u32 = 4;
    const VEC4: u32 = 5;
    const IMAGE: u32 = 6;
    const SAMPLED_IMAGE: u32 = 7;
    const PTR_UNIFORM_SAMPLED: u32 = 8;
    const PTR_INPUT_VEC2: u32 = 9;
    const PTR_INPUT_VEC4: u32 = 10;
    const PTR_OUTPUT_VEC4: u32 = 11;
    const TEXTURE: u32 = 12;
    const IN_UV: u32 = 13;
    const IN_COLOR: u32 = 14;
    const OUT_COLOR: u32 = 15;
    const MAIN: u32 = 16;
    const LABEL: u32 = 17;
    const TEXTURE_VALUE: u32 = 18;
    const UV: u32 = 19;
    const SAMPLE: u32 = 20;
    const COLOR: u32 = 21;
    const RESULT: u32 = 22;

    let mut module = Module::new(23);
    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_FRAGMENT, MAIN],
        "main",
        &[IN_UV, IN_COLOR, OUT_COLOR],
    );
    module.instruction(
        OP_EXECUTION_MODE,
        &[MAIN, EXECUTION_ORIGIN_UPPER_LEFT],
    );

    module.instruction(OP_DECORATE, &[TEXTURE, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[TEXTURE, DECORATION_BINDING, 0]);
    module.instruction(OP_DECORATE, &[IN_UV, DECORATION_LOCATION, 0]);
    module.instruction(OP_DECORATE, &[IN_COLOR, DECORATION_LOCATION, 1]);
    module.instruction(OP_DECORATE, &[OUT_COLOR, DECORATION_LOCATION, 0]);

    module.instruction(OP_TYPE_VOID, &[VOID]);
    module.instruction(OP_TYPE_FUNCTION, &[FUNCTION_TYPE, VOID]);
    module.instruction(OP_TYPE_FLOAT, &[FLOAT, 32]);
    module.instruction(OP_TYPE_VECTOR, &[VEC2, FLOAT, 2]);
    module.instruction(OP_TYPE_VECTOR, &[VEC4, FLOAT, 4]);
    module.instruction(OP_TYPE_IMAGE, &[IMAGE, FLOAT, 1, 0, 0, 0, 1, 0]);
    module.instruction(OP_TYPE_SAMPLED_IMAGE, &[SAMPLED_IMAGE, IMAGE]);
    module.instruction(
        OP_TYPE_POINTER,
        &[PTR_UNIFORM_SAMPLED, STORAGE_UNIFORM_CONSTANT, SAMPLED_IMAGE],
    );
    module.instruction(OP_TYPE_POINTER, &[PTR_INPUT_VEC2, STORAGE_INPUT, VEC2]);
    module.instruction(OP_TYPE_POINTER, &[PTR_INPUT_VEC4, STORAGE_INPUT, VEC4]);
    module.instruction(OP_TYPE_POINTER, &[PTR_OUTPUT_VEC4, STORAGE_OUTPUT, VEC4]);

    module.instruction(
        OP_VARIABLE,
        &[PTR_UNIFORM_SAMPLED, TEXTURE, STORAGE_UNIFORM_CONSTANT],
    );
    module.instruction(OP_VARIABLE, &[PTR_INPUT_VEC2, IN_UV, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[PTR_INPUT_VEC4, IN_COLOR, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[PTR_OUTPUT_VEC4, OUT_COLOR, STORAGE_OUTPUT]);

    module.instruction(OP_FUNCTION, &[VOID, MAIN, 0, FUNCTION_TYPE]);
    module.instruction(OP_LABEL, &[LABEL]);
    module.instruction(OP_LOAD, &[SAMPLED_IMAGE, TEXTURE_VALUE, TEXTURE]);
    module.instruction(OP_LOAD, &[VEC2, UV, IN_UV]);
    module.instruction(
        OP_IMAGE_SAMPLE_IMPLICIT_LOD,
        &[VEC4, SAMPLE, TEXTURE_VALUE, UV],
    );
    module.instruction(OP_LOAD, &[VEC4, COLOR, IN_COLOR]);
    module.instruction(OP_FMUL, &[VEC4, RESULT, SAMPLE, COLOR]);
    module.instruction(OP_STORE, &[OUT_COLOR, RESULT]);
    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modules_have_valid_headers() {
        for words in [ui_vertex(), ui_fragment()] {
            assert_eq!(words[0], 0x0723_0203);
            assert_eq!(words[1], 0x0001_0000);
            assert!(words.len() > 20);
            assert!(words[3] > 1);
        }
    }
}