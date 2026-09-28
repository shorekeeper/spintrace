//! SPIR-V modules for spin state initialization and one Bloch field step.
//!
//! Both shaders use a local size of sixty four. The host pads every storage
//! buffer to that width, so the entry points need no divergent bounds branch.
//!
//! Spin parameters occupy two vec4 values per record. State occupies one vec4.
//! The evolve shader receives RF, duration, gyromagnetic ratio and gradient in
//! a thirty two byte push constant block.
//!
//! The numerical path matches sim::cpu: half relaxation, exact Rodrigues
//! rotation around the complete effective angular frequency, then the remaining
//! half relaxation. A zero frequency uses an epsilon denominator while retaining
//! a zero angle, which leaves the relaxed vector unchanged.

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
const OP_FNEGATE: u16 = 127;
const OP_IADD: u16 = 128;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_IMUL: u16 = 132;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;

const CAPABILITY_SHADER: u32 = 1;
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_GLSL450: u32 = 1;
const EXECUTION_GL_COMPUTE: u32 = 5;
const EXECUTION_LOCAL_SIZE: u32 = 17;

const STORAGE_INPUT: u32 = 1;
const STORAGE_PUSH_CONSTANT: u32 = 9;
const STORAGE_BUFFER: u32 = 12;

const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const BUILT_IN_GLOBAL_INVOCATION_ID: u32 = 28;

const GLSL_SIN: u32 = 13;
const GLSL_COS: u32 = 14;
const GLSL_EXP: u32 = 27;
const GLSL_SQRT: u32 = 31;
const GLSL_FMAX: u32 = 40;

pub const WORKGROUP_SIZE: u32 = 64;

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

    fn ext_result(
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

fn extract(module: &mut Module, result_type: u32, value: u32, component: u32) -> u32 {
    module.result(OP_COMPOSITE_EXTRACT, result_type, &[value, component])
}

fn unary(module: &mut Module, opcode: u16, result_type: u32, value: u32) -> u32 {
    module.result(opcode, result_type, &[value])
}

fn binary(
    module: &mut Module,
    opcode: u16,
    result_type: u32,
    left: u32,
    right: u32,
) -> u32 {
    module.result(opcode, result_type, &[left, right])
}

pub fn reset() -> Vec<u32> {
    let mut module = Module::new();

    let void = module.id();
    let function_type = module.id();
    let uint = module.id();
    let float = module.id();
    let uvec3 = module.id();
    let vec4 = module.id();
    let spin_param = module.id();
    let params_array = module.id();
    let params_block = module.id();
    let state_array = module.id();
    let state_block = module.id();
    let ptr_input_uvec3 = module.id();
    let ptr_storage_params = module.id();
    let ptr_storage_states = module.id();
    let ptr_storage_float = module.id();
    let ptr_storage_vec4 = module.id();
    let global_id = module.id();
    let params = module.id();
    let states = module.id();
    let zero_u = module.id();
    let three_u = module.id();
    let zero_f = module.id();
    let main = module.id();

    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_GL_COMPUTE, main],
        "main",
        &[global_id],
    );
    module.instruction(
        OP_EXECUTION_MODE,
        &[main, EXECUTION_LOCAL_SIZE, WORKGROUP_SIZE, 1, 1],
    );

    module.instruction(
        OP_DECORATE,
        &[global_id, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID],
    );
    module.instruction(OP_DECORATE, &[params_array, DECORATION_ARRAY_STRIDE, 32]);
    module.instruction(OP_MEMBER_DECORATE, &[spin_param, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_MEMBER_DECORATE, &[spin_param, 1, DECORATION_OFFSET, 16]);
    module.instruction(OP_DECORATE, &[params_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[params_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[state_array, DECORATION_ARRAY_STRIDE, 16]);
    module.instruction(OP_DECORATE, &[state_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[state_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[params, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[params, DECORATION_BINDING, 0]);
    module.instruction(OP_DECORATE, &[params, DECORATION_NON_WRITABLE]);
    module.instruction(OP_DECORATE, &[states, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_BINDING, 1]);

    module.instruction(OP_TYPE_VOID, &[void]);
    module.instruction(OP_TYPE_FUNCTION, &[function_type, void]);
    module.instruction(OP_TYPE_INT, &[uint, 32, 0]);
    module.instruction(OP_TYPE_FLOAT, &[float, 32]);
    module.instruction(OP_TYPE_VECTOR, &[uvec3, uint, 3]);
    module.instruction(OP_TYPE_VECTOR, &[vec4, float, 4]);
    module.instruction(OP_TYPE_STRUCT, &[spin_param, vec4, vec4]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[params_array, spin_param]);
    module.instruction(OP_TYPE_STRUCT, &[params_block, params_array]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[state_array, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[state_block, state_array]);
    module.instruction(OP_TYPE_POINTER, &[ptr_input_uvec3, STORAGE_INPUT, uvec3]);
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_params, STORAGE_BUFFER, params_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_states, STORAGE_BUFFER, state_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_float, STORAGE_BUFFER, float],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_vec4, STORAGE_BUFFER, vec4],
    );

    module.instruction(OP_VARIABLE, &[ptr_input_uvec3, global_id, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_storage_params, params, STORAGE_BUFFER]);
    module.instruction(OP_VARIABLE, &[ptr_storage_states, states, STORAGE_BUFFER]);

    module.instruction(OP_CONSTANT, &[uint, zero_u, 0]);
    module.instruction(OP_CONSTANT, &[uint, three_u, 3]);
    module.instruction(OP_CONSTANT, &[float, zero_f, 0.0f32.to_bits()]);

    module.instruction(OP_FUNCTION, &[void, main, 0, function_type]);
    let label = module.id();
    module.instruction(OP_LABEL, &[label]);

    let invocation = module.result(OP_LOAD, uvec3, &[global_id]);
    let index = extract(&mut module, uint, invocation, 0);
    let m0_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_float,
        &[params, zero_u, index, zero_u, three_u],
    );
    let m0 = module.result(OP_LOAD, float, &[m0_ptr]);
    let state_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[states, zero_u, index],
    );
    let value = module.result(
        OP_COMPOSITE_CONSTRUCT,
        vec4,
        &[zero_f, zero_f, m0, zero_f],
    );
    module.instruction(OP_STORE, &[state_ptr, value]);
    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

pub fn evolve() -> Vec<u32> {
    let mut module = Module::new();

    let glsl = module.id();
    let void = module.id();
    let function_type = module.id();
    let uint = module.id();
    let float = module.id();
    let uvec3 = module.id();
    let vec4 = module.id();
    let spin_param = module.id();
    let params_array = module.id();
    let params_block = module.id();
    let state_array = module.id();
    let state_block = module.id();
    let push_block = module.id();
    let ptr_input_uvec3 = module.id();
    let ptr_storage_params = module.id();
    let ptr_storage_states = module.id();
    let ptr_storage_vec4 = module.id();
    let ptr_push_block = module.id();
    let ptr_push_vec4 = module.id();
    let global_id = module.id();
    let params = module.id();
    let states = module.id();
    let push = module.id();
    let zero_u = module.id();
    let one_u = module.id();
    let zero_f = module.id();
    let one_f = module.id();
    let half_f = module.id();
    let epsilon_f = module.id();
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
        &[main, EXECUTION_LOCAL_SIZE, WORKGROUP_SIZE, 1, 1],
    );

    module.instruction(
        OP_DECORATE,
        &[global_id, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID],
    );
    module.instruction(OP_DECORATE, &[params_array, DECORATION_ARRAY_STRIDE, 32]);
    module.instruction(OP_MEMBER_DECORATE, &[spin_param, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_MEMBER_DECORATE, &[spin_param, 1, DECORATION_OFFSET, 16]);
    module.instruction(OP_DECORATE, &[params_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[params_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[state_array, DECORATION_ARRAY_STRIDE, 16]);
    module.instruction(OP_DECORATE, &[state_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[state_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[push_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[push_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_MEMBER_DECORATE, &[push_block, 1, DECORATION_OFFSET, 16]);
    module.instruction(OP_DECORATE, &[params, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[params, DECORATION_BINDING, 0]);
    module.instruction(OP_DECORATE, &[params, DECORATION_NON_WRITABLE]);
    module.instruction(OP_DECORATE, &[states, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_BINDING, 1]);

    module.instruction(OP_TYPE_VOID, &[void]);
    module.instruction(OP_TYPE_FUNCTION, &[function_type, void]);
    module.instruction(OP_TYPE_INT, &[uint, 32, 0]);
    module.instruction(OP_TYPE_FLOAT, &[float, 32]);
    module.instruction(OP_TYPE_VECTOR, &[uvec3, uint, 3]);
    module.instruction(OP_TYPE_VECTOR, &[vec4, float, 4]);
    module.instruction(OP_TYPE_STRUCT, &[spin_param, vec4, vec4]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[params_array, spin_param]);
    module.instruction(OP_TYPE_STRUCT, &[params_block, params_array]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[state_array, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[state_block, state_array]);
    module.instruction(OP_TYPE_STRUCT, &[push_block, vec4, vec4]);
    module.instruction(OP_TYPE_POINTER, &[ptr_input_uvec3, STORAGE_INPUT, uvec3]);
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_params, STORAGE_BUFFER, params_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_states, STORAGE_BUFFER, state_block],
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

    module.instruction(OP_VARIABLE, &[ptr_input_uvec3, global_id, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_storage_params, params, STORAGE_BUFFER]);
    module.instruction(OP_VARIABLE, &[ptr_storage_states, states, STORAGE_BUFFER]);
    module.instruction(OP_VARIABLE, &[ptr_push_block, push, STORAGE_PUSH_CONSTANT]);

    module.instruction(OP_CONSTANT, &[uint, zero_u, 0]);
    module.instruction(OP_CONSTANT, &[uint, one_u, 1]);
    module.instruction(OP_CONSTANT, &[float, zero_f, 0.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, one_f, 1.0f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, half_f, 0.5f32.to_bits()]);
    module.instruction(OP_CONSTANT, &[float, epsilon_f, 1.0e-20f32.to_bits()]);

    module.instruction(OP_FUNCTION, &[void, main, 0, function_type]);
    let label = module.id();
    module.instruction(OP_LABEL, &[label]);

    let invocation = module.result(OP_LOAD, uvec3, &[global_id]);
    let index = extract(&mut module, uint, invocation, 0);

    let position_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[params, zero_u, index, zero_u],
    );
    let rates_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[params, zero_u, index, one_u],
    );
    let state_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[states, zero_u, index],
    );
    let field_ptr =
        module.result(OP_ACCESS_CHAIN, ptr_push_vec4, &[push, zero_u]);
    let gradient_ptr =
        module.result(OP_ACCESS_CHAIN, ptr_push_vec4, &[push, one_u]);

    let position = module.result(OP_LOAD, vec4, &[position_ptr]);
    let rates = module.result(OP_LOAD, vec4, &[rates_ptr]);
    let state = module.result(OP_LOAD, vec4, &[state_ptr]);
    let field = module.result(OP_LOAD, vec4, &[field_ptr]);
    let gradient = module.result(OP_LOAD, vec4, &[gradient_ptr]);

    let px = extract(&mut module, float, position, 0);
    let py = extract(&mut module, float, position, 1);
    let pz = extract(&mut module, float, position, 2);
    let m0 = extract(&mut module, float, position, 3);
    let inv_t1 = extract(&mut module, float, rates, 0);
    let inv_t2 = extract(&mut module, float, rates, 1);
    let offset = extract(&mut module, float, rates, 2);
    let mx = extract(&mut module, float, state, 0);
    let my = extract(&mut module, float, state, 1);
    let mz = extract(&mut module, float, state, 2);
    let b1x = extract(&mut module, float, field, 0);
    let b1y = extract(&mut module, float, field, 1);
    let dt = extract(&mut module, float, field, 2);
    let gamma = extract(&mut module, float, field, 3);
    let gx = extract(&mut module, float, gradient, 0);
    let gy = extract(&mut module, float, gradient, 1);
    let gz = extract(&mut module, float, gradient, 2);

    let half_dt = binary(&mut module, OP_FMUL, float, dt, half_f);
    let t2_exp_arg = binary(&mut module, OP_FMUL, float, half_dt, inv_t2);
    let t2_exp_arg = unary(&mut module, OP_FNEGATE, float, t2_exp_arg);
    let transverse =
        module.ext_result(float, glsl, GLSL_EXP, &[t2_exp_arg]);
    let t1_exp_arg = binary(&mut module, OP_FMUL, float, half_dt, inv_t1);
    let t1_exp_arg = unary(&mut module, OP_FNEGATE, float, t1_exp_arg);
    let longitudinal =
        module.ext_result(float, glsl, GLSL_EXP, &[t1_exp_arg]);

    let mx_half = binary(&mut module, OP_FMUL, float, mx, transverse);
    let my_half = binary(&mut module, OP_FMUL, float, my, transverse);
    let mz_delta = binary(&mut module, OP_FSUB, float, mz, m0);
    let mz_decay = binary(&mut module, OP_FMUL, float, mz_delta, longitudinal);
    let mz_half = binary(&mut module, OP_FADD, float, m0, mz_decay);

    let gx_x = binary(&mut module, OP_FMUL, float, gx, px);
    let gy_y = binary(&mut module, OP_FMUL, float, gy, py);
    let gz_z = binary(&mut module, OP_FMUL, float, gz, pz);
    let gradient_xy = binary(&mut module, OP_FADD, float, gx_x, gy_y);
    let gradient_dot = binary(&mut module, OP_FADD, float, gradient_xy, gz_z);
    let gradient_omega =
        binary(&mut module, OP_FMUL, float, gamma, gradient_dot);
    let omega_z =
        binary(&mut module, OP_FADD, float, offset, gradient_omega);
    let omega_x = binary(&mut module, OP_FMUL, float, gamma, b1x);
    let omega_y = binary(&mut module, OP_FMUL, float, gamma, b1y);

    let omega_x2 = binary(&mut module, OP_FMUL, float, omega_x, omega_x);
    let omega_y2 = binary(&mut module, OP_FMUL, float, omega_y, omega_y);
    let omega_z2 = binary(&mut module, OP_FMUL, float, omega_z, omega_z);
    let omega_xy = binary(&mut module, OP_FADD, float, omega_x2, omega_y2);
    let omega_sq = binary(&mut module, OP_FADD, float, omega_xy, omega_z2);
    let omega_mag =
        module.ext_result(float, glsl, GLSL_SQRT, &[omega_sq]);
    let safe_mag =
        module.ext_result(float, glsl, GLSL_FMAX, &[omega_mag, epsilon_f]);

    let axis_x = binary(&mut module, OP_FDIV, float, omega_x, safe_mag);
    let axis_y = binary(&mut module, OP_FDIV, float, omega_y, safe_mag);
    let axis_z = binary(&mut module, OP_FDIV, float, omega_z, safe_mag);
    let angle = binary(&mut module, OP_FMUL, float, omega_mag, dt);
    let sine = module.ext_result(float, glsl, GLSL_SIN, &[angle]);
    let cosine = module.ext_result(float, glsl, GLSL_COS, &[angle]);
    let one_minus_cosine =
        binary(&mut module, OP_FSUB, float, one_f, cosine);

    let cross_x_a = binary(&mut module, OP_FMUL, float, my_half, axis_z);
    let cross_x_b = binary(&mut module, OP_FMUL, float, mz_half, axis_y);
    let cross_x = binary(&mut module, OP_FSUB, float, cross_x_a, cross_x_b);
    let cross_y_a = binary(&mut module, OP_FMUL, float, mz_half, axis_x);
    let cross_y_b = binary(&mut module, OP_FMUL, float, mx_half, axis_z);
    let cross_y = binary(&mut module, OP_FSUB, float, cross_y_a, cross_y_b);
    let cross_z_a = binary(&mut module, OP_FMUL, float, mx_half, axis_y);
    let cross_z_b = binary(&mut module, OP_FMUL, float, my_half, axis_x);
    let cross_z = binary(&mut module, OP_FSUB, float, cross_z_a, cross_z_b);

    let projection_x = binary(&mut module, OP_FMUL, float, mx_half, axis_x);
    let projection_y = binary(&mut module, OP_FMUL, float, my_half, axis_y);
    let projection_z = binary(&mut module, OP_FMUL, float, mz_half, axis_z);
    let projection_xy =
        binary(&mut module, OP_FADD, float, projection_x, projection_y);
    let projection =
        binary(&mut module, OP_FADD, float, projection_xy, projection_z);
    let projection_curve =
        binary(&mut module, OP_FMUL, float, projection, one_minus_cosine);

    let mx_cos = binary(&mut module, OP_FMUL, float, mx_half, cosine);
    let mx_sin = binary(&mut module, OP_FMUL, float, cross_x, sine);
    let mx_axis = binary(&mut module, OP_FMUL, float, axis_x, projection_curve);
    let mx_rot_a = binary(&mut module, OP_FADD, float, mx_cos, mx_sin);
    let mx_rot = binary(&mut module, OP_FADD, float, mx_rot_a, mx_axis);

    let my_cos = binary(&mut module, OP_FMUL, float, my_half, cosine);
    let my_sin = binary(&mut module, OP_FMUL, float, cross_y, sine);
    let my_axis = binary(&mut module, OP_FMUL, float, axis_y, projection_curve);
    let my_rot_a = binary(&mut module, OP_FADD, float, my_cos, my_sin);
    let my_rot = binary(&mut module, OP_FADD, float, my_rot_a, my_axis);

    let mz_cos = binary(&mut module, OP_FMUL, float, mz_half, cosine);
    let mz_sin = binary(&mut module, OP_FMUL, float, cross_z, sine);
    let mz_axis = binary(&mut module, OP_FMUL, float, axis_z, projection_curve);
    let mz_rot_a = binary(&mut module, OP_FADD, float, mz_cos, mz_sin);
    let mz_rot = binary(&mut module, OP_FADD, float, mz_rot_a, mz_axis);

    let mx_out = binary(&mut module, OP_FMUL, float, mx_rot, transverse);
    let my_out = binary(&mut module, OP_FMUL, float, my_rot, transverse);
    let mz_out_delta = binary(&mut module, OP_FSUB, float, mz_rot, m0);
    let mz_out_decay =
        binary(&mut module, OP_FMUL, float, mz_out_delta, longitudinal);
    let mz_out = binary(&mut module, OP_FADD, float, m0, mz_out_decay);

    let output = module.result(
        OP_COMPOSITE_CONSTRUCT,
        vec4,
        &[mx_out, my_out, mz_out, zero_f],
    );
    module.instruction(OP_STORE, &[state_ptr, output]);
    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

/// Maps one spin to its weighted complex receiver contribution.
///
/// The output is a vec4 so every reduction buffer keeps sixteen byte elements.
/// Only x and y carry data; z and w are reserved for later coil channels and
/// auxiliary statistics.
pub fn signal_map() -> Vec<u32> {
    let mut module = Module::new();

    let void = module.id();
    let function_type = module.id();
    let uint = module.id();
    let float = module.id();
    let uvec3 = module.id();
    let vec4 = module.id();
    let spin_param = module.id();
    let params_array = module.id();
    let params_block = module.id();
    let state_array = module.id();
    let state_block = module.id();
    let sum_array = module.id();
    let sum_block = module.id();
    let ptr_input_uvec3 = module.id();
    let ptr_storage_params = module.id();
    let ptr_storage_states = module.id();
    let ptr_storage_sums = module.id();
    let ptr_storage_vec4 = module.id();
    let global_id = module.id();
    let params = module.id();
    let states = module.id();
    let sums = module.id();
    let zero_u = module.id();
    let one_u = module.id();
    let three_u = module.id();
    let zero_f = module.id();
    let main = module.id();

    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_GL_COMPUTE, main],
        "main",
        &[global_id],
    );
    module.instruction(
        OP_EXECUTION_MODE,
        &[main, EXECUTION_LOCAL_SIZE, WORKGROUP_SIZE, 1, 1],
    );

    module.instruction(
        OP_DECORATE,
        &[global_id, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID],
    );
    module.instruction(OP_DECORATE, &[params_array, DECORATION_ARRAY_STRIDE, 32]);
    module.instruction(OP_MEMBER_DECORATE, &[spin_param, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_MEMBER_DECORATE, &[spin_param, 1, DECORATION_OFFSET, 16]);
    module.instruction(OP_DECORATE, &[params_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[params_block, 0, DECORATION_OFFSET, 0]);

    module.instruction(OP_DECORATE, &[state_array, DECORATION_ARRAY_STRIDE, 16]);
    module.instruction(OP_DECORATE, &[state_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[state_block, 0, DECORATION_OFFSET, 0]);

    module.instruction(OP_DECORATE, &[sum_array, DECORATION_ARRAY_STRIDE, 16]);
    module.instruction(OP_DECORATE, &[sum_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[sum_block, 0, DECORATION_OFFSET, 0]);

    module.instruction(OP_DECORATE, &[params, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[params, DECORATION_BINDING, 0]);
    module.instruction(OP_DECORATE, &[params, DECORATION_NON_WRITABLE]);
    module.instruction(OP_DECORATE, &[states, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[states, DECORATION_BINDING, 1]);
    module.instruction(OP_DECORATE, &[states, DECORATION_NON_WRITABLE]);
    module.instruction(OP_DECORATE, &[sums, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[sums, DECORATION_BINDING, 2]);

    module.instruction(OP_TYPE_VOID, &[void]);
    module.instruction(OP_TYPE_FUNCTION, &[function_type, void]);
    module.instruction(OP_TYPE_INT, &[uint, 32, 0]);
    module.instruction(OP_TYPE_FLOAT, &[float, 32]);
    module.instruction(OP_TYPE_VECTOR, &[uvec3, uint, 3]);
    module.instruction(OP_TYPE_VECTOR, &[vec4, float, 4]);
    module.instruction(OP_TYPE_STRUCT, &[spin_param, vec4, vec4]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[params_array, spin_param]);
    module.instruction(OP_TYPE_STRUCT, &[params_block, params_array]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[state_array, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[state_block, state_array]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[sum_array, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[sum_block, sum_array]);

    module.instruction(OP_TYPE_POINTER, &[ptr_input_uvec3, STORAGE_INPUT, uvec3]);
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_params, STORAGE_BUFFER, params_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_states, STORAGE_BUFFER, state_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_sums, STORAGE_BUFFER, sum_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_vec4, STORAGE_BUFFER, vec4],
    );

    module.instruction(OP_VARIABLE, &[ptr_input_uvec3, global_id, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_storage_params, params, STORAGE_BUFFER]);
    module.instruction(OP_VARIABLE, &[ptr_storage_states, states, STORAGE_BUFFER]);
    module.instruction(OP_VARIABLE, &[ptr_storage_sums, sums, STORAGE_BUFFER]);

    module.instruction(OP_CONSTANT, &[uint, zero_u, 0]);
    module.instruction(OP_CONSTANT, &[uint, one_u, 1]);
    module.instruction(OP_CONSTANT, &[uint, three_u, 3]);
    module.instruction(OP_CONSTANT, &[float, zero_f, 0.0f32.to_bits()]);

    module.instruction(OP_FUNCTION, &[void, main, 0, function_type]);
    let label = module.id();
    module.instruction(OP_LABEL, &[label]);

    let invocation = module.result(OP_LOAD, uvec3, &[global_id]);
    let index = extract(&mut module, uint, invocation, 0);
    let rates_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[params, zero_u, index, one_u],
    );
    let state_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[states, zero_u, index],
    );
    let sum_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[sums, zero_u, index],
    );

    let rates = module.result(OP_LOAD, vec4, &[rates_ptr]);
    let state = module.result(OP_LOAD, vec4, &[state_ptr]);
    let weight = extract(&mut module, float, rates, 3);
    let mx = extract(&mut module, float, state, 0);
    let my = extract(&mut module, float, state, 1);
    let real = binary(&mut module, OP_FMUL, float, mx, weight);
    let imaginary = binary(&mut module, OP_FMUL, float, my, weight);
    let value = module.result(
        OP_COMPOSITE_CONSTRUCT,
        vec4,
        &[real, imaginary, zero_f, zero_f],
    );

    module.instruction(OP_STORE, &[sum_ptr, value]);
    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

/// Adds adjacent vec4 values into the next reduction level.
///
/// Input counts are powers of two and the host stops while at least one full
/// workgroup remains. Every invocation therefore reads and writes valid storage
/// without a bounds branch.
pub fn signal_reduce() -> Vec<u32> {
    let mut module = Module::new();

    let void = module.id();
    let function_type = module.id();
    let uint = module.id();
    let float = module.id();
    let uvec3 = module.id();
    let vec4 = module.id();
    let source_array = module.id();
    let source_block = module.id();
    let target_array = module.id();
    let target_block = module.id();
    let ptr_input_uvec3 = module.id();
    let ptr_storage_source = module.id();
    let ptr_storage_target = module.id();
    let ptr_storage_vec4 = module.id();
    let global_id = module.id();
    let source = module.id();
    let target = module.id();
    let zero_u = module.id();
    let one_u = module.id();
    let two_u = module.id();
    let main = module.id();

    module.instruction(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    module.instruction(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
    module.string_instruction(
        OP_ENTRY_POINT,
        &[EXECUTION_GL_COMPUTE, main],
        "main",
        &[global_id],
    );
    module.instruction(
        OP_EXECUTION_MODE,
        &[main, EXECUTION_LOCAL_SIZE, WORKGROUP_SIZE, 1, 1],
    );

    module.instruction(
        OP_DECORATE,
        &[global_id, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID],
    );
    module.instruction(OP_DECORATE, &[source_array, DECORATION_ARRAY_STRIDE, 16]);
    module.instruction(OP_DECORATE, &[source_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[source_block, 0, DECORATION_OFFSET, 0]);
    module.instruction(OP_DECORATE, &[target_array, DECORATION_ARRAY_STRIDE, 16]);
    module.instruction(OP_DECORATE, &[target_block, DECORATION_BLOCK]);
    module.instruction(OP_MEMBER_DECORATE, &[target_block, 0, DECORATION_OFFSET, 0]);

    module.instruction(OP_DECORATE, &[source, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[source, DECORATION_BINDING, 0]);
    module.instruction(OP_DECORATE, &[source, DECORATION_NON_WRITABLE]);
    module.instruction(OP_DECORATE, &[target, DECORATION_DESCRIPTOR_SET, 0]);
    module.instruction(OP_DECORATE, &[target, DECORATION_BINDING, 1]);

    module.instruction(OP_TYPE_VOID, &[void]);
    module.instruction(OP_TYPE_FUNCTION, &[function_type, void]);
    module.instruction(OP_TYPE_INT, &[uint, 32, 0]);
    module.instruction(OP_TYPE_FLOAT, &[float, 32]);
    module.instruction(OP_TYPE_VECTOR, &[uvec3, uint, 3]);
    module.instruction(OP_TYPE_VECTOR, &[vec4, float, 4]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[source_array, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[source_block, source_array]);
    module.instruction(OP_TYPE_RUNTIME_ARRAY, &[target_array, vec4]);
    module.instruction(OP_TYPE_STRUCT, &[target_block, target_array]);

    module.instruction(OP_TYPE_POINTER, &[ptr_input_uvec3, STORAGE_INPUT, uvec3]);
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_source, STORAGE_BUFFER, source_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_target, STORAGE_BUFFER, target_block],
    );
    module.instruction(
        OP_TYPE_POINTER,
        &[ptr_storage_vec4, STORAGE_BUFFER, vec4],
    );

    module.instruction(OP_VARIABLE, &[ptr_input_uvec3, global_id, STORAGE_INPUT]);
    module.instruction(OP_VARIABLE, &[ptr_storage_source, source, STORAGE_BUFFER]);
    module.instruction(OP_VARIABLE, &[ptr_storage_target, target, STORAGE_BUFFER]);

    module.instruction(OP_CONSTANT, &[uint, zero_u, 0]);
    module.instruction(OP_CONSTANT, &[uint, one_u, 1]);
    module.instruction(OP_CONSTANT, &[uint, two_u, 2]);

    module.instruction(OP_FUNCTION, &[void, main, 0, function_type]);
    let label = module.id();
    module.instruction(OP_LABEL, &[label]);

    let invocation = module.result(OP_LOAD, uvec3, &[global_id]);
    let index = extract(&mut module, uint, invocation, 0);
    let left_index = module.result(OP_IMUL, uint, &[index, two_u]);
    let right_index = module.result(OP_IADD, uint, &[left_index, one_u]);

    let left_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[source, zero_u, left_index],
    );
    let right_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[source, zero_u, right_index],
    );
    let target_ptr = module.result(
        OP_ACCESS_CHAIN,
        ptr_storage_vec4,
        &[target, zero_u, index],
    );

    let left = module.result(OP_LOAD, vec4, &[left_ptr]);
    let right = module.result(OP_LOAD, vec4, &[right_ptr]);
    let sum = module.result(OP_FADD, vec4, &[left, right]);
    module.instruction(OP_STORE, &[target_ptr, sum]);

    module.instruction(OP_RETURN, &[]);
    module.instruction(OP_FUNCTION_END, &[]);
    module.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_modules_have_spirv_headers() {
        for words in [reset(), evolve(), signal_map(), signal_reduce()] {
            assert_eq!(words[0], 0x0723_0203);
            assert_eq!(words[1], 0x0001_0300);
            assert_eq!(words[3] as usize > 1, true);
            assert!(words.len() > 40);
        }
    }
}