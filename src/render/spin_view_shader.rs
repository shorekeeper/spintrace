//! SPIR-V modules for direct spin visualization.
//!
//! Phase bins are cleared with vkCmdFillBuffer and populated by one compute
//! invocation per uploaded spin. Every real record carries one in the reserved
//! state component and every padded record carries zero, so the atomic add can
//! remain branch free.
//!
//! The vector vertex shader reads magnetization directly from a storage buffer.
//! One instance emits one line from the centre of a grid cell to the transverse
//! magnetization endpoint. The histogram vertex shader reads the phase bins and
//! expands six fixed corner vertices into one bar per bin.

use std::f32::consts::{PI, TAU};

const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_CONVERT_F_TO_U: u16 = 109;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_FNEGATE: u16 = 127;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_IMUL: u16 = 132;
const OP_FMUL: u16 = 133;
const OP_UDIV: u16 = 134;
const OP_FDIV: u16 = 136;
const OP_UMOD: u16 = 137;
const OP_ATOMIC_IADD: u16 = 234;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;

const CAPABILITY_SHADER: u32 = 1;
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_GLSL450: u32 = 1;
const EXECUTION_VERTEX: u32 = 0;
const EXECUTION_FRAGMENT: u32 = 4;
const EXECUTION_GL_COMPUTE: u32 = 5;
const EXECUTION_ORIGIN_UPPER_LEFT: u32 = 7;
const EXECUTION_LOCAL_SIZE: u32 = 17;

const STORAGE_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_INPUT: u32 = 1;
const STORAGE_OUTPUT: u32 = 3;
const STORAGE_PUSH_CONSTANT: u32 = 9;
const STORAGE_BUFFER: u32 = 12;

const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_LOCATION: u32 = 30;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

const BUILT_IN_POSITION: u32 = 0;
const BUILT_IN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILT_IN_INSTANCE_INDEX: u32 = 43;

const GLSL_SIN: u32 = 13;
const GLSL_COS: u32 = 14;
const GLSL_ATAN2: u32 = 25;
const GLSL_SQRT: u32 = 31;
const GLSL_FMIN: u32 = 37;
const GLSL_FMAX: u32 = 40;

pub const VISUAL_SPINS: usize = 512;
pub const PHASE_BINS: usize = 24;
pub const VECTOR_INSTANCES: u32 = 24;
pub const HISTOGRAM_GROUPS: u32 = (VISUAL_SPINS as u32) / 64;

struct Module {
    words: Vec<u32>,
    next_id: u32,
}

impl Module {
    fn new() -> Module {
        Module {
            words: vec![0x0723_0203, 0x0001_0300, 0, 0, 0],
            next_id: 1,
        }
    }

    fn id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn instruction(&mut self, opcode: u16, operands: &[u32]) {
        self.words
            .push(((operands.len() as u32 + 1) << 16) | opcode as u32);
        self.words.extend_from_slice(operands);
    }

    fn string_instruction(
        &mut self,
        opcode: u16,
        prefix: &[u32],
        text: &str,
        suffix: &[u32],
    ) {
        let mut operands =
            Vec::with_capacity(prefix.len() + suffix.len() + text.len() / 4 + 1);
        operands.extend_from_slice(prefix);

        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        for chunk in bytes.chunks_exact(4) {
            operands.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        operands.extend_from_slice(suffix);
        self.instruction(opcode, &operands);
    }

    fn result(&mut self, opcode: u16, result_type: u32, operands: &[u32]) -> u32 {
        let result = self.id();
        let mut all = Vec::with_capacity(operands.len() + 2);
        all.push(result_type);
        all.push(result);
        all.extend_from_slice(operands);
        self.instruction(opcode, &all);
        result
    }

    fn ext(
        &mut self,
        result_type: u32,
        set: u32,
        instruction: u32,
        operands: &[u32],
    ) -> u32 {
        let mut all = Vec::with_capacity(operands.len() + 2);
        all.push(set);
        all.push(instruction);
        all.extend_from_slice(operands);
        self.result(OP_EXT_INST, result_type, &all)
    }

    fn finish(mut self) -> Vec<u32> {
        self.words[3] = self.next_id;
        self.words
    }
}

fn extract(module: &mut Module, ty: u32, value: u32, component: u32) -> u32 {
    module.result(OP_COMPOSITE_EXTRACT, ty, &[value, component])
}

fn unary(module: &mut Module, opcode: u16, ty: u32, value: u32) -> u32 {
    module.result(opcode, ty, &[value])
}

fn binary(
    module: &mut Module,
    opcode: u16,
    ty: u32,
    left: u32,
    right: u32,
) -> u32 {
    module.result(opcode, ty, &[left, right])
}

pub fn phase_histogram() -> Vec<u32> {
    let mut module = Module::new();

    let glsl = module.id();
    let void = module.id();
    let function_type = module.id();
    let uint = module.id();
    let float = module.id();
    let uvec3 = module.id();
    let vec4 = module.id();
    let state_array = module.id();
    let state_block = module.id();
    let bin_array = module.id();
    let bin_block = module.id();
    let ptr_input_uvec3 = module.id();
    let ptr_storage_state_block = module.id();
    let ptr_storage_bin_block = module.id();
    let ptr_storage_vec4 = module.id();
    let ptr_storage_uint = module.id();
    let global_id = module.id();
    let states = module.id();
    let bins = module.id();
    let zero_u = module.id();
    let one_u = module.id();
    let scope_device = module.id();
    let semantics_relaxed = module.id();
    let pi = module.id();
    let bin_scale = module.id();
    let main = module.id();

    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.string_instruction(OP_EXT_INST_IMPORT, &[glsl], "GLSL.std.450", &[]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_GL_COMPUTE, main],
        "main",
        &[global_id],
    );
    module.instruction(
        OP_EXECUTION_MODE,
        &[main, EXECUTION_LOCAL_SIZE, 64, 1, 1],
    );

    module.instruction(
        OP_DECORATE,
        &[global_id, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID],
    );
    module.instruction(OP_DECORATE, &[state_array, DECORATION_ARRAY_STRIDE, 16]);
    module.instruction(OP_DECORATE, &[state_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[state_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[bin_array, DECORATION_ARRAY_STRIDE, 4]);
    module.instruction(OP_DECORATE, &[bin_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[bin_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_BINDING, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_NON_WRITABLE]);
    module.instruction(OP_DECORATE, &[bins, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[bins, DECORATION_BINDING, 1]);

    module.instruction(OP_TYPE_VOID, &[void]);
    module.instruction(OP_TYPE_FUNCTION, &[function_type, void]);
    module.instruction(OP_TYPE_INT, &[uint, 32, 0]);
    module.instruction(OP_TYPE_FLOAT, &[float, 32]);
    module.instruction(OP_TYPE_VECTOR, &[uvec3, uint, 3]);
    module.instruction(OP_TYPE_VECTOR, &[vec4, float, 4]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[state_array, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[state_block, state_array]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[bin_array, uint]);
    module.instruction(OP_TYPE_STRUCT, &[bin_block, bin_array]);
    module.instruction(OP_TYPE_POINTER, &[ptr_input_uvec3, STORAGE_INPUT, uvec3]);
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_state_block, STORAGE_BUFFER, state_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_bin_block, STORAGE_BUFFER, bin_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_vec4, STORAGE_BUFFER, vec4],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_uint, STORAGE_BUFFER, uint],
    );

    module.instruction(OP_VARIABLE, &[ptr_input_uvec3, global_id, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_storage_state_block, states, STORAGE_BUFFER]);
    module.instruction(OP_VARIABLE, &[ptr_storage_bin_block, bins, STORAGE_BUFFER]);

    module.instruction(OP_CONSTANT, &[uint, zero_u, 0]);
    module.instruction(OP_CONSTANT, &[uint, one_u, 1]);
    module.instruction(OP_CONSTANT, &[uint, scope_device, 1]);
    module.instruction(OP_CONSTANT, &[uint, semantics_relaxed, 0]);
    module.instruction(OP_CONSTANT, &[float, pi, PI.to_bits()]);
    module.instruction(
        OP_CONSTANT,
        &[float, bin_scale, (23.999f32 / TAU).to_bits()],
    );

    module.instruction(OP_FUNCTION, &[void, main, 0, function_type]);
    let label = module.id();
    module.instruction(OP_LABEL, &[label]);

    let invocation = module.result(OP_LOAD, uvec3, &[global_id]);
    let index = extract(&mut module, uint, invocation, 0);
    let state_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[states, zero_u, index],
    );
    let state = module.result(OP_LOAD, vec4, &[state_ptr]);
    let mx = extract(&mut module, float, state, 0);
    let my = extract(&mut module, float, state, 1);
    let active_f = extract(&mut module, float, state, 3);
    let active = module.result(OP_CONVERT_F_TO_U, uint, &[active_f]);

    let phase = module.ext(float, glsl, GLSL_ATAN2, &[my, mx]);
    let shifted = binary(&mut module, OP_FADD, float, phase, pi);
    let scaled = binary(&mut module, OP_FMUL, float, shifted, bin_scale);
    let bin = module.result(OP_CONVERT_F_TO_U, uint, &[scaled]);
    let bin_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_uint,
        &[bins, zero_u, bin],
    );
    let _old = module.result(
        OP_ATOMIC_IADD,
        uint,
        &[bin_ptr, scope_device, semantics_relaxed, active],
    );

    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

pub fn vectors_vertex() -> Vec<u32> {
    let mut module = Module::new();

    let glsl = module.id();
    let void = module.id();
    let function_type = module.id();
    let uint = module.id();
    let float = module.id();
    let vec2 = module.id();
    let vec4 = module.id();
    let state_array = module.id();
    let state_block = module.id();
    let per_vertex = module.id();
    let push_block = module.id();
    let ptr_input_vec2 = module.id();
    let ptr_input_uint = module.id();
    let ptr_output_vec4 = module.id();
    let ptr_output_per_vertex = module.id();
    let ptr_storage_state_block = module.id();
    let ptr_storage_vec4 = module.id();
    let ptr_push_block = module.id();
    let ptr_push_vec4 = module.id();
    let corner = module.id();
    let instance = module.id();
    let out_color = module.id();
    let out_per_vertex = module.id();
    let states = module.id();
    let push = module.id();

    let zero_u = module.id();
    let one_u = module.id();
    let six_u = module.id();
    let twenty_one_u = module.id();
    let zero_f = module.id();
    let one_f = module.id();
    let two_f = module.id();
    let half_f = module.id();
    let six_f = module.id();
    let four_f = module.id();
    let vector_fraction = module.id();
    let color_base = module.id();
    let color_scale = module.id();
    let minus_half = module.id();
    let rotation = module.id();
    let epsilon = module.id();
    let main = module.id();

    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.string_instruction(OP_EXT_INST_IMPORT, &[glsl], "GLSL.std.450", &[]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_VERTEX, main],
        "main",
        &[corner, instance, out_color, out_per_vertex],
    );

    module.instruction(OP_DECORATE, &[corner, DECORATION_LOCATION, 0]);
    module.instruction(
        OP_DECORATE,
        &[instance, DECORATION_BUILT_IN, BUILT_IN_INSTANCE_INDEX],
    );
    module.instruction(OP_DECORATE, &[out_color, DECORATION_LOCATION, 0]);
    module.instruction(OP_DECORATE, &[per_vertex, DECORATION_BLOCK]);
    module.instruction(
        OP_MEMBER_DECORATE,
        &[per_vertex, 0, DECORATION_BUILT_IN, BUILT_IN_POSITION],
    );
    module.instruction(OP_DECORATE, &[state_array, DECORATION_ARRAY_STRIDE, 16]);
    module.instruction(OP_DECORATE, &[state_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[state_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_BINDING, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_NON_WRITABLE]);
    module.instruction(OP_DECORATE, &[push_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[push_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_MEMBER_DECORATE, &[push_block, 1, DECORATION_OFFSET, 16]);

    module.instruction(OP_TYPE_VOID, &[void]);
    module.instruction(OP_TYPE_FUNCTION, &[function_type, void]);
    module.instruction(OP_TYPE_INT, &[uint, 32, 0]);
    module.instruction(OP_TYPE_FLOAT, &[float, 32]);
    module.instruction(OP_TYPE_VECTOR, &[vec2, float, 2]);
    module.instruction(OP_TYPE_VECTOR, &[vec4, float, 4]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[state_array, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[state_block, state_array]);
    module.instruction(OP_TYPE_STRUCT, &[per_vertex, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[push_block, vec4, vec4]);

    module.instruction(OP_TYPE_POINTER, &[ptr_input_vec2, STORAGE_INPUT, vec2]);
    module.instruction(OP_TYPE_POINTER, &[ptr_input_uint, STORAGE_INPUT, uint]);
    module.instruction(OP_TYPE_POINTER, &[ptr_output_vec4, STORAGE_OUTPUT, vec4]);
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_output_per_vertex, STORAGE_OUTPUT, per_vertex],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_state_block, STORAGE_BUFFER, state_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_vec4, STORAGE_BUFFER, vec4],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_push_block, STORAGE_PUSH_CONSTANT, push_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_push_vec4, STORAGE_PUSH_CONSTANT, vec4],
    );

    module.instruction(OP_VARIABLE, &[ptr_input_vec2, corner, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_input_uint, instance, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_output_vec4, out_color, STORAGE_OUTPUT]);
    module.instruction(
        OP_VARIABLE,
        &[ptr_output_per_vertex, out_per_vertex, STORAGE_OUTPUT],
    );
    module.instruction(
        OP_VARIABLE,
        &[ptr_storage_state_block, states, STORAGE_BUFFER],
    );
    module.instruction(OP_VARIABLE, &[ptr_push_block, push, STORAGE_PUSH_CONSTANT]);

    module.instruction(OP_CONSTANT, &[uint, zero_u, 0]);
    module.instruction(OP_CONSTANT, &[uint, one_u, 1]);
    module.instruction(OP_CONSTANT, &[uint, six_u, 6]);
    module.instruction(OP_CONSTANT, &[uint, twenty_one_u, 21]);
    module.instruction(OP_CONSTANT, &[float, zero_f, 0.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, one_f, 1.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, two_f, 2.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, half_f, 0.5f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, six_f, 6.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, four_f, 4.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, vector_fraction, 0.38f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, color_base, 0.55f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, color_scale, 0.45f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, minus_half, (-0.5f32).to_bits()]);
    module.instruction(OP_CONSTANT, &[float, rotation, 0.866_025_4f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, epsilon, 1.0e-12f32.to_bits()]);

    module.instruction(OP_FUNCTION, &[void, main, 0, function_type]);
    let label = module.id();
    module.instruction(OP_LABEL, &[label]);

    let instance_value = module.result(OP_LOAD, uint, &[instance]);
    let state_index =
        module.result(OP_IMUL, uint, &[instance_value, twenty_one_u]);
    let state_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[states, zero_u, state_index],
    );
    let state = module.result(OP_LOAD, vec4, &[state_ptr]);
    let mx = extract(&mut module, float, state, 0);
    let my = extract(&mut module, float, state, 1);

    let rect_ptr =
        module.result(OP_ACCESS_CHAIN, ptr_push_vec4, &[push, zero_u]);
    let params_ptr =
        module.result(OP_ACCESS_CHAIN, ptr_push_vec4, &[push, one_u]);
    let rect = module.result(OP_LOAD, vec4, &[rect_ptr]);
    let params = module.result(OP_LOAD, vec4, &[params_ptr]);

    let rect_x = extract(&mut module, float, rect, 0);
    let rect_y = extract(&mut module, float, rect, 1);
    let rect_w = extract(&mut module, float, rect, 2);
    let rect_h = extract(&mut module, float, rect, 3);
    let viewport_w = extract(&mut module, float, params, 0);
    let viewport_h = extract(&mut module, float, params, 1);
    let user_scale = extract(&mut module, float, params, 2);

    let column = module.result(OP_UMOD, uint, &[instance_value, six_u]);
    let row = module.result(OP_UDIV, uint, &[instance_value, six_u]);
    let column_f = module.result(OP_CONVERT_U_TO_F, float, &[column]);
    let row_f = module.result(OP_CONVERT_U_TO_F, float, &[row]);
    let column_center = binary(&mut module, OP_FADD, float, column_f, half_f);
    let row_center = binary(&mut module, OP_FADD, float, row_f, half_f);
    let cell_w = binary(&mut module, OP_FDIV, float, rect_w, six_f);
    let cell_h = binary(&mut module, OP_FDIV, float, rect_h, four_f);
    let center_x_offset =
        binary(&mut module, OP_FMUL, float, column_center, cell_w);
    let center_y_offset =
        binary(&mut module, OP_FMUL, float, row_center, cell_h);
    let center_x = binary(&mut module, OP_FADD, float, rect_x, center_x_offset);
    let center_y = binary(&mut module, OP_FADD, float, rect_y, center_y_offset);

    let cell_min = module.ext(float, glsl, GLSL_FMIN, &[cell_w, cell_h]);
    let vector_length =
        binary(&mut module, OP_FMUL, float, cell_min, vector_fraction);
    let vector_length =
        binary(&mut module, OP_FMUL, float, vector_length, user_scale);

    let corner_value = module.result(OP_LOAD, vec2, &[corner]);
    let factor = extract(&mut module, float, corner_value, 0);
    let x_offset = binary(&mut module, OP_FMUL, float, mx, vector_length);
    let x_offset = binary(&mut module, OP_FMUL, float, x_offset, factor);
    let negative_my = unary(&mut module, OP_FNEGATE, float, my);
    let y_offset = binary(&mut module, OP_FMUL, float, negative_my, vector_length);
    let y_offset = binary(&mut module, OP_FMUL, float, y_offset, factor);
    let pixel_x = binary(&mut module, OP_FADD, float, center_x, x_offset);
    let pixel_y = binary(&mut module, OP_FADD, float, center_y, y_offset);

    let x_normal = binary(&mut module, OP_FDIV, float, pixel_x, viewport_w);
    let y_normal = binary(&mut module, OP_FDIV, float, pixel_y, viewport_h);
    let x_scaled = binary(&mut module, OP_FMUL, float, x_normal, two_f);
    let y_scaled = binary(&mut module, OP_FMUL, float, y_normal, two_f);
    let ndc_x = binary(&mut module, OP_FSUB, float, x_scaled, one_f);
    let ndc_y = binary(&mut module, OP_FSUB, float, y_scaled, one_f);

    let position = module.result(
        OP_COMPOSITE_CONSTRUCT,
        vec4,
        &[ndc_x, ndc_y, zero_f, one_f],
    );
    let position_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_output_vec4,
        &[out_per_vertex, zero_u],
    );
    module.instruction(OP_STORE, &[position_ptr, position]);

    let mx2 = binary(&mut module, OP_FMUL, float, mx, mx);
    let my2 = binary(&mut module, OP_FMUL, float, my, my);
    let magnitude2 = binary(&mut module, OP_FADD, float, mx2, my2);
    let magnitude = module.ext(float, glsl, GLSL_SQRT, &[magnitude2]);
    let safe = module.ext(float, glsl, GLSL_FMAX, &[magnitude, epsilon]);
    let nx = binary(&mut module, OP_FDIV, float, mx, safe);
    let ny = binary(&mut module, OP_FDIV, float, my, safe);

    let red_delta = binary(&mut module, OP_FMUL, float, color_scale, nx);
    let red = binary(&mut module, OP_FADD, float, color_base, red_delta);
    let green_x = binary(&mut module, OP_FMUL, float, minus_half, nx);
    let green_y = binary(&mut module, OP_FMUL, float, rotation, ny);
    let green_mix = binary(&mut module, OP_FADD, float, green_x, green_y);
    let green_delta =
        binary(&mut module, OP_FMUL, float, color_scale, green_mix);
    let green =
        binary(&mut module, OP_FADD, float, color_base, green_delta);
    let blue_x = binary(&mut module, OP_FMUL, float, minus_half, nx);
    let blue_y = binary(&mut module, OP_FMUL, float, rotation, ny);
    let blue_mix = binary(&mut module, OP_FSUB, float, blue_x, blue_y);
    let blue_delta =
        binary(&mut module, OP_FMUL, float, color_scale, blue_mix);
    let blue = binary(&mut module, OP_FADD, float, color_base, blue_delta);

    let color = module.result(
        OP_COMPOSITE_CONSTRUCT,
        vec4,
        &[red, green, blue, one_f],
    );
    module.instruction(OP_STORE, &[out_color, color]);
    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

pub fn histogram_vertex() -> Vec<u32> {
    let mut module = Module::new();

    let glsl = module.id();
    let void = module.id();
    let function_type = module.id();
    let uint = module.id();
    let float = module.id();
    let vec2 = module.id();
    let vec4 = module.id();
    let bin_array = module.id();
    let bin_block = module.id();
    let per_vertex = module.id();
    let push_block = module.id();
    let ptr_input_vec2 = module.id();
    let ptr_input_uint = module.id();
    let ptr_output_vec4 = module.id();
    let ptr_output_per_vertex = module.id();
    let ptr_storage_bin_block = module.id();
    let ptr_storage_uint = module.id();
    let ptr_push_block = module.id();
    let ptr_push_vec4 = module.id();

    let corner = module.id();
    let instance = module.id();
    let out_color = module.id();
    let out_per_vertex = module.id();
    let bins = module.id();
    let push = module.id();

    let zero_u = module.id();
    let one_u = module.id();
    let zero_f = module.id();
    let one_f = module.id();
    let two_f = module.id();
    let half_f = module.id();
    let bins_f = module.id();
    let spins_f = module.id();
    let bar_width = module.id();
    let phase_step = module.id();
    let pi = module.id();
    let color_base = module.id();
    let color_scale = module.id();
    let minus_half = module.id();
    let rotation = module.id();
    let main = module.id();

    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.string_instruction(OP_EXT_INST_IMPORT, &[glsl], "GLSL.std.450", &[]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_VERTEX, main],
        "main",
        &[corner, instance, out_color, out_per_vertex],
    );

    module.instruction(OP_DECORATE, &[corner, DECORATION_LOCATION, 0]);
    module.instruction(
        OP_DECORATE,
        &[instance, DECORATION_BUILT_IN, BUILT_IN_INSTANCE_INDEX],
    );
    module.instruction(OP_DECORATE, &[out_color, DECORATION_LOCATION, 0]);
    module.instruction(OP_DECORATE, &[per_vertex, DECORATION_BLOCK]);
    module.instruction(
        OP_MEMBER_DECORATE,
        &[per_vertex, 0, DECORATION_BUILT_IN, BUILT_IN_POSITION],
    );
    module.instruction(OP_DECORATE, &[bin_array, DECORATION_ARRAY_STRIDE, 4]);
    module.instruction(OP_DECORATE, &[bin_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[bin_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[bins, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[bins, DECORATION_BINDING, 1]);
    module.instruction(OP_DECORATE, &[bins, DECORATION_NON_WRITABLE]);
    module.instruction(OP_DECORATE, &[push_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[push_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_MEMBER_DECORATE, &[push_block, 1, DECORATION_OFFSET, 16]);

    module.instruction(OP_TYPE_VOID, &[void]);
    module.instruction(OP_TYPE_FUNCTION, &[function_type, void]);
    module.instruction(OP_TYPE_INT, &[uint, 32, 0]);
    module.instruction(OP_TYPE_FLOAT, &[float, 32]);
    module.instruction(OP_TYPE_VECTOR, &[vec2, float, 2]);
    module.instruction(OP_TYPE_VECTOR, &[vec4, float, 4]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[bin_array, uint]);
    module.instruction(OP_TYPE_STRUCT, &[bin_block, bin_array]);
    module.instruction(OP_TYPE_STRUCT, &[per_vertex, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[push_block, vec4, vec4]);

    module.instruction(OP_TYPE_POINTER, &[ptr_input_vec2, STORAGE_INPUT, vec2]);
    module.instruction(OP_TYPE_POINTER, &[ptr_input_uint, STORAGE_INPUT, uint]);
    module.instruction(OP_TYPE_POINTER, &[ptr_output_vec4, STORAGE_OUTPUT, vec4]);
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_output_per_vertex, STORAGE_OUTPUT, per_vertex],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_bin_block, STORAGE_BUFFER, bin_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_uint, STORAGE_BUFFER, uint],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_push_block, STORAGE_PUSH_CONSTANT, push_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_push_vec4, STORAGE_PUSH_CONSTANT, vec4],
    );

    module.instruction(OP_VARIABLE, &[ptr_input_vec2, corner, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_input_uint, instance, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_output_vec4, out_color, STORAGE_OUTPUT]);
    module.instruction(
        OP_VARIABLE,
        &[ptr_output_per_vertex, out_per_vertex, STORAGE_OUTPUT],
    );
    module.instruction(OP_VARIABLE, &[ptr_storage_bin_block, bins, STORAGE_BUFFER]);
    module.instruction(OP_VARIABLE, &[ptr_push_block, push, STORAGE_PUSH_CONSTANT]);

    module.instruction(OP_CONSTANT, &[uint, zero_u, 0]);
    module.instruction(OP_CONSTANT, &[uint, one_u, 1]);
    module.instruction(OP_CONSTANT, &[float, zero_f, 0.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, one_f, 1.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, two_f, 2.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, half_f, 0.5f32.to_bits()]);
    module.instruction(
        OP_CONSTANT,
        &[float, bins_f, (PHASE_BINS as f32).to_bits()],
    );
    module.instruction(
        OP_CONSTANT,
        &[float, spins_f, (VISUAL_SPINS as f32).to_bits()],
    );
    module.instruction(OP_CONSTANT, &[float, bar_width, 0.88f32.to_bits()]);
    module.instruction(
        OP_CONSTANT,
        &[float, phase_step, (TAU / PHASE_BINS as f32).to_bits()],
    );
    module.instruction(OP_CONSTANT, &[float, pi, PI.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, color_base, 0.55f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, color_scale, 0.45f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, minus_half, (-0.5f32).to_bits()]);
    module.instruction(OP_CONSTANT, &[float, rotation, 0.866_025_4f32.to_bits()]);

    module.instruction(OP_FUNCTION, &[void, main, 0, function_type]);
    let label = module.id();
    module.instruction(OP_LABEL, &[label]);

    let bin_index = module.result(OP_LOAD, uint, &[instance]);
    let bin_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_uint,
        &[bins, zero_u, bin_index],
    );
    let count = module.result(OP_LOAD, uint, &[bin_ptr]);
    let count_f = module.result(OP_CONVERT_U_TO_F, float, &[count]);
    let fraction = binary(&mut module, OP_FDIV, float, count_f, spins_f);
    let height_fraction = module.ext(float, glsl, GLSL_SQRT, &[fraction]);

    let rect_ptr =
        module.result(OP_ACCESS_CHAIN, ptr_push_vec4, &[push, zero_u]);
    let params_ptr =
        module.result(OP_ACCESS_CHAIN, ptr_push_vec4, &[push, one_u]);
    let rect = module.result(OP_LOAD, vec4, &[rect_ptr]);
    let params = module.result(OP_LOAD, vec4, &[params_ptr]);
    let rect_x = extract(&mut module, float, rect, 0);
    let rect_y = extract(&mut module, float, rect, 1);
    let rect_w = extract(&mut module, float, rect, 2);
    let rect_h = extract(&mut module, float, rect, 3);
    let viewport_w = extract(&mut module, float, params, 0);
    let viewport_h = extract(&mut module, float, params, 1);

    let corner_value = module.result(OP_LOAD, vec2, &[corner]);
    let corner_x = extract(&mut module, float, corner_value, 0);
    let corner_y = extract(&mut module, float, corner_value, 1);
    let instance_f = module.result(OP_CONVERT_U_TO_F, float, &[bin_index]);
    let cell_w = binary(&mut module, OP_FDIV, float, rect_w, bins_f);
    let corner_x = binary(&mut module, OP_FMUL, float, corner_x, bar_width);
    let x_cell = binary(&mut module, OP_FADD, float, instance_f, corner_x);
    let x_offset = binary(&mut module, OP_FMUL, float, x_cell, cell_w);
    let pixel_x = binary(&mut module, OP_FADD, float, rect_x, x_offset);

    let height = binary(&mut module, OP_FMUL, float, rect_h, height_fraction);
    let bottom = binary(&mut module, OP_FADD, float, rect_y, rect_h);
    let top = binary(&mut module, OP_FSUB, float, bottom, height);
    let y_offset = binary(&mut module, OP_FMUL, float, corner_y, height);
    let pixel_y = binary(&mut module, OP_FADD, float, top, y_offset);

    let x_normal = binary(&mut module, OP_FDIV, float, pixel_x, viewport_w);
    let y_normal = binary(&mut module, OP_FDIV, float, pixel_y, viewport_h);
    let x_scaled = binary(&mut module, OP_FMUL, float, x_normal, two_f);
    let y_scaled = binary(&mut module, OP_FMUL, float, y_normal, two_f);
    let ndc_x = binary(&mut module, OP_FSUB, float, x_scaled, one_f);
    let ndc_y = binary(&mut module, OP_FSUB, float, y_scaled, one_f);

    let position = module.result(
        OP_COMPOSITE_CONSTRUCT,
        vec4,
        &[ndc_x, ndc_y, zero_f, one_f],
    );
    let position_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_output_vec4,
        &[out_per_vertex, zero_u],
    );
    module.instruction(OP_STORE, &[position_ptr, position]);

    let phase_index =
        binary(&mut module, OP_FADD, float, instance_f, half_f);
    let phase =
        binary(&mut module, OP_FMUL, float, phase_index, phase_step);
    let phase = binary(&mut module, OP_FSUB, float, phase, pi);
    let cosine = module.ext(float, glsl, GLSL_COS, &[phase]);
    let sine = module.ext(float, glsl, GLSL_SIN, &[phase]);

    let red_delta =
        binary(&mut module, OP_FMUL, float, color_scale, cosine);
    let red = binary(&mut module, OP_FADD, float, color_base, red_delta);
    let green_x =
        binary(&mut module, OP_FMUL, float, minus_half, cosine);
    let green_y =
        binary(&mut module, OP_FMUL, float, rotation, sine);
    let green_mix =
        binary(&mut module, OP_FADD, float, green_x, green_y);
    let green_delta =
        binary(&mut module, OP_FMUL, float, color_scale, green_mix);
    let green =
        binary(&mut module, OP_FADD, float, color_base, green_delta);
    let blue_x =
        binary(&mut module, OP_FMUL, float, minus_half, cosine);
    let blue_y =
        binary(&mut module, OP_FMUL, float, rotation, sine);
    let blue_mix =
        binary(&mut module, OP_FSUB, float, blue_x, blue_y);
    let blue_delta =
        binary(&mut module, OP_FMUL, float, color_scale, blue_mix);
    let blue =
        binary(&mut module, OP_FADD, float, color_base, blue_delta);

    let color = module.result(
        OP_COMPOSITE_CONSTRUCT,
        vec4,
        &[red, green, blue, one_f],
    );
    module.instruction(OP_STORE, &[out_color, color]);
    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

pub fn solid_fragment() -> Vec<u32> {
    let mut module = Module::new();

    let void = module.id();
    let function_type = module.id();
    let float = module.id();
    let vec4 = module.id();
    let ptr_input_vec4 = module.id();
    let ptr_output_vec4 = module.id();
    let input_color = module.id();
    let output_color = module.id();
    let main = module.id();

    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_FRAGMENT, main],
        "main",
        &[input_color, output_color],
    );
    module.instruction(
        OP_EXECUTION_MODE,
        &[main, EXECUTION_ORIGIN_UPPER_LEFT],
    );
    module.instruction(OP_DECORATE, &[input_color, DECORATION_LOCATION, 0]);
    module.instruction(OP_DECORATE, &[output_color, DECORATION_LOCATION, 0]);

    module.instruction(OP_TYPE_VOID, &[void]);
    module.instruction(OP_TYPE_FUNCTION, &[function_type, void]);
    module.instruction(OP_TYPE_FLOAT, &[float, 32]);
    module.instruction(OP_TYPE_VECTOR, &[vec4, float, 4]);
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_input_vec4, STORAGE_INPUT, vec4],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_output_vec4, STORAGE_OUTPUT, vec4],
    );
    module.instruction(
        OP_VARIABLE,
        &[ptr_input_vec4, input_color, STORAGE_INPUT],
    );
    module.instruction(
        OP_VARIABLE,
        &[ptr_output_vec4, output_color, STORAGE_OUTPUT],
    );

    module.instruction(OP_FUNCTION, &[void, main, 0, function_type]);
    let label = module.id();
    module.instruction(OP_LABEL, &[label]);
    let color = module.result(OP_LOAD, vec4, &[input_color]);
    module.instruction(OP_STORE, &[output_color, color]);
    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visualization_modules_have_spirv_headers() {
        for words in [
            phase_histogram(),
            vectors_vertex(),
            histogram_vertex(),
            solid_fragment(),
        ] {
            assert_eq!(words[0], 0x0723_0203);
            assert_eq!(words[1], 0x0001_0300);
            assert!(words[3] > 1);
            assert!(words.len() > 30);
        }
    }

    #[test]
    fn visual_constants_cover_complete_workgroups() {
        assert_eq!(VISUAL_SPINS % 64, 0);
        assert_eq!(HISTOGRAM_GROUPS * 64, VISUAL_SPINS as u32);
        assert_eq!(VECTOR_INSTANCES, 24);
        assert_eq!(PHASE_BINS, 24);
    }
}