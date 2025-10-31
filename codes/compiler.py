import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
import copy
from collections import deque, namedtuple
import tempfile
import os

# ==============
# EMULATOR INTEGRATION
# ==============

MASK32 = 0xFFFFFFFF

def u32(x): 
    return x & MASK32

def s32(x):  # interpret x as signed 32
    x &= MASK32
    return x if x < 0x80000000 else x - 0x100000000

class ALU:
    # ALUControl encodings (3 bits): 000=ADD, 001=SUB, 010=AND, 011=MUL, 100=OR, 101=SLT
    def compute(self, A: int, B: int, ALUControl: int):
        A = u32(A); B = u32(B)
        Cout = 0; OverFlow = 0
        
        if ALUControl == 0b000:       # ADD
            full = (A + B)
            Result = u32(full)
            Cout = (full >> 32) & 1
            # Overflow when adding same sign gives different sign
            a31 = (A >> 31) & 1; b31 = (B >> 31) & 1; r31 = (Result >> 31) & 1
            OverFlow = 1 if (a31 == b31 and r31 != a31) else 0
            
        elif ALUControl == 0b001:     # SUB -> A + (~B + 1)
            B2 = u32(~B + 1)
            full = (A + B2)
            Result = u32(full)
            Cout = (full >> 32) & 1
            a31 = (A >> 31) & 1; b31 = (B >> 31) & 1; r31 = (Result >> 31) & 1
            # Overflow when subtracting different sign yields wrong sign
            OverFlow = 1 if (a31 != b31 and r31 != a31) else 0
            
        elif ALUControl == 0b010:     # AND
            Result = u32(A & B)
            
        elif ALUControl == 0b011:     # MUL
            full = A * B
            Result = u32(full)
            # For multiplication, we can detect overflow if result exceeds 32 bits
            OverFlow = 1 if (full >> 32) != 0 else 0
            
        elif ALUControl == 0b100:     # OR
            Result = u32(A | B)
            
        elif ALUControl == 0b101:     # SLT (signed)
            Result = 1 if s32(A) < s32(B) else 0
            
        else:
            Result = 0
            
        Zero = 1 if Result == 0 else 0
        Negative = (Result >> 31) & 1
        Carry = Cout if ((ALUControl == 0b000) or (ALUControl == 0b001)) else 0
        
        return Result, Carry, OverFlow, Zero, Negative

class ALUDecoder:
    # RV32I + M extension mapping -> 000=ADD, 001=SUB, 010=AND, 011=MUL, 100=OR, 101=SLT
    def decode(self, ALUOp: int, funct3: int, funct7: int, op: int) -> int:
        if ALUOp == 0b00:         # loads/stores/addi default to ADD
            return 0b000
        elif ALUOp == 0b01:       # branches (BEQ uses SUB for compare)
            return 0b001
            
        # R-type or some I-type with ALU operations
        if ALUOp == 0b10:
            if funct3 == 0b000:
                if funct7 == 0x01:    # M extension - MUL
                    return 0b011
                else:
                    # check if SUB (funct7 = 0x20) vs ADD (funct7=0x00)
                    return 0b001 if (funct7 == 0x20) else 0b000
            elif funct3 == 0b111:
                return 0b010    # AND
            elif funct3 == 0b110:
                return 0b100    # OR
            elif funct3 == 0b010:
                return 0b101    # SLT
                
        return 0b000

class ControlUnit:
    def __init__(self):
        self.alu_decoder = ALUDecoder()
        
    # returns: RegWrite, ImmSrc(2), ALUSrc, MemWrite, ResultSrc, Branch, ALUControl(3), ALUOp(2)
    def decode(self, Op: int, funct3: int, funct7: int):
        RegWrite = ImmSrc = ALUSrc = MemWrite = ResultSrc = Branch = 0
        ALUOp = 0b00
        
        # Opcodes (RV32I)
        LOAD     = 0b0000011  # LW
        STORE    = 0b0100011  # SW
        R_TYPE   = 0b0110011
        I_TYPE   = 0b0010011  # ADDI/ORI/ANDI (we'll treat as ADDI in ALU)
        BRANCH   = 0b1100011  # BEQ only here
        
        if Op in (LOAD, R_TYPE, I_TYPE): 
            RegWrite = 1
            
        if Op == STORE: 
            ImmSrc = 0b01
        elif Op == BRANCH: 
            ImmSrc = 0b10
        else: 
            ImmSrc = 0b00
            
        if Op in (LOAD, STORE, I_TYPE): 
            ALUSrc = 1
            
        if Op == STORE: 
            MemWrite = 1
            
        if Op == LOAD: 
            ResultSrc = 1
            
        if Op == BRANCH: 
            Branch = 1
            
        if Op == R_TYPE: 
            ALUOp = 0b10
        elif Op == BRANCH: 
            ALUOp = 0b01
        else: 
            ALUOp = 0b00
            
        ALUControl = self.alu_decoder.decode(ALUOp, funct3, funct7, Op)
        
        return RegWrite, ImmSrc, ALUSrc, MemWrite, ResultSrc, Branch, ALUControl, ALUOp

class DataMemory:
    def __init__(self, depth=1024):
        self.mem = [0] * depth
        
    def access(self, rst: int, WE: int, A: int, WD: int) -> int:
        A_idx = (A >> 2) & 0x3FF  # word addressed
        if rst == 0:
            return 0
        if WE:
            self.mem[A_idx] = u32(WD)
        return self.mem[A_idx]

class InstructionMemory:
    def __init__(self, depth=1024, instructions=None):
        self.mem = [0] * depth
        if instructions:
            self.load_instructions(instructions)
        
    def load_instructions(self, instructions):
        for i, instr in enumerate(instructions):
            if i < len(self.mem):
                if isinstance(instr, str):
                    # Remove comments and parse hex
                    hex_part = instr.split(';')[0].strip()
                    if hex_part.startswith('0x'):
                        self.mem[i] = int(hex_part, 16) & MASK32
                    else:
                        self.mem[i] = int(hex_part, 16) & MASK32
                else:
                    self.mem[i] = instr & MASK32
            
    def read(self, rst: int, A: int) -> int:
        if rst == 0:
            return 0
        return self.mem[(A >> 2) & 0x3FF]

class Mux2:
    def select(self, a: int, b: int, s: int) -> int:
        return a if s == 0 else b

class PC_Module:
    def __init__(self):
        self.PCF = 0  # current PC (fetch stage)
        
    def update(self, clk: int, rst: int, PC_Next: int):
        if clk == 1:
            if rst == 0:
                self.PCF = 0
            else:
                self.PCF = u32(PC_Next)

class PCAdder:
    def add(self, a: int, b: int) -> int:
        return u32(a + b)

class RegisterFile:
    def __init__(self):
        self.regs = [0] * 32
        self.write_log = []
        
    def write(self, WE3: int, A3: int, WD3: int):
        if WE3 and A3 != 0:
            self.regs[A3] = u32(WD3)
            self.write_log.append(f"x{A3:02d} <= 0x{self.regs[A3]:08X} ({s32(self.regs[A3])})")
            
    def read(self, rst: int, A1: int, A2: int):
        RD1 = 0 if rst == 0 else self.regs[A1]
        RD2 = 0 if rst == 0 else self.regs[A2]
        return RD1, RD2
        
    def get_register_state(self):
        state = {}
        for i in range(32):
            if self.regs[i] != 0:  # Only show non-zero registers
                state[f'x{i:02d}'] = {
                    'hex': f'0x{self.regs[i]:08X}',
                    'decimal': s32(self.regs[i])
                }
        return state

class SignExtend:
    def extend(self, instr: int, ImmSrc: int) -> int:
        # I-type (Imm[31:20])
        if ImmSrc == 0b00:
            imm = (instr >> 20) & 0xFFF
            if imm & 0x800:
                imm |= ~0xFFF  # sign extend
            return u32(imm)
        # S-type (Imm[31:25|11:7])
        elif ImmSrc == 0b01:
            imm = ((instr >> 25) & 0x7F) << 5 | ((instr >> 7) & 0x1F)
            if imm & 0x800:
                imm |= ~0xFFF
            return u32(imm)
        # B-type (Imm[31|7|30:25|11:8] << 1)
        elif ImmSrc == 0b10:
            imm = ((instr >> 31) & 0x1) << 12
            imm |= ((instr >> 7) & 0x1) << 11
            imm |= ((instr >> 25) & 0x3F) << 5
            imm |= ((instr >> 8) & 0xF) << 1
            if imm & 0x1000:
                imm |= ~0x1FFF
            return u32(imm)
        return 0

class SimpleRV32I:
    def __init__(self, instructions=None):
        self.imem = InstructionMemory(instructions=instructions)
        self.dmem = DataMemory()
        self.rf = RegisterFile()
        self.alu = ALU()
        self.ctrl = ControlUnit()
        self.signext = SignExtend()
        self.pc = PC_Module()
        self.pc_adder = PCAdder()
        self.mux2 = Mux2()
        
    def step(self, rst: int = 1):
        clk = 1  # single rising edge per step
        
        # FETCH
        instr = self.imem.read(rst, self.pc.PCF)
        pc_plus_4 = self.pc_adder.add(self.pc.PCF, 4)
        
        # DECODE
        Op = instr & 0x7F
        rd = (instr >> 7) & 0x1F
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        funct7 = (instr >> 25) & 0x7F
        
        RegWrite, ImmSrc, ALUSrc, MemWrite, ResultSrc, Branch, ALUControl, ALUOp = \
            self.ctrl.decode(Op, funct3, funct7)
            
        RD1, RD2 = self.rf.read(rst, rs1, rs2)
        imm_ext = self.signext.extend(instr, ImmSrc)
        
        # EXECUTE
        SrcB = self.mux2.select(RD2, imm_ext, ALUSrc)
        alu_res, Carry, OverFlow, Zero, Negative = self.alu.compute(RD1, SrcB, ALUControl)
        
        # BRANCH TARGET
        pc_target = self.pc_adder.add(self.pc.PCF, imm_ext)
        
        # MEMORY
        read_data = self.dmem.access(rst, MemWrite, alu_res, RD2)
        
        # WRITEBACK
        result_w = self.mux2.select(alu_res, read_data, ResultSrc)
        self.rf.write(RegWrite, rd, result_w)
        
        # NEXT PC
        PCSrc = 1 if (Branch and Zero) else 0
        next_pc = self.mux2.select(pc_plus_4, pc_target, PCSrc)
        self.pc.update(clk, rst, next_pc)
        
        # Halt on all-zero instruction
        return instr != 0
        
    def run(self, max_steps=100, rst=1):
        steps = 0
        while steps < max_steps:
            cont = self.step(rst=rst)
            steps += 1
            if not cont:
                break
        return self.rf.get_register_state(), self.rf.write_log

# ==========================================================
# AST NODES
# ==========================================================

class ASTNode: 
    pass

class Assign(ASTNode):
    def __init__(self, dest, expr):
        self.dest, self.expr = dest, expr

class Print(ASTNode):
    def __init__(self, var):
        self.var = var

class BinOp(ASTNode):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right

class Var(ASTNode):
    def __init__(self, name):
        self.name = name

class Const(ASTNode):
    def __init__(self, value):
        self.value = int(value)

# ==========================================================
# PARSER (tiny C-like subset)
# ==========================================================

def _is_int(s):
    return isinstance(s, str) and (s.isdigit() or (s.startswith('-') and s[1:].isdigit()))

def parse_source(src):
    lines = [ln.strip() for ln in src.splitlines() if ln.strip() and not ln.strip().startswith('//')]
    stmts = []
    
    for ln in lines:
        if ln.startswith('print(') and ln.endswith(')'):
            inner = ln[len('print('):-1].strip()
            stmts.append(Print(inner))
            continue
            
        m = re.match(r'([a-zA-Z_]\w*)\s*=\s*(.+)', ln)
        if not m:
            raise SyntaxError(f"Cannot parse line: {ln}")
            
        dest = m.group(1)
        rhs = m.group(2)
        
        bm = re.match(r'(.+)\s*([+\-*])\s*(.+)', rhs)
        if bm:
            left = bm.group(1).strip()
            op = bm.group(2)
            right = bm.group(3).strip()
            left_node = Const(left) if _is_int(left) else Var(left)
            right_node = Const(right) if _is_int(right) else Var(right)
            expr = BinOp(op, left_node, right_node)
        else:
            token = rhs.strip()
            expr = Const(token) if _is_int(token) else Var(token)
            
        stmts.append(Assign(dest, expr))
        
    return stmts

# ==========================================================
# REGISTER ALLOCATION
# ==========================================================

VAR_REG_POOL = [f'x{i}' for i in range(1, 17)]  # x1 to x16
TEMP_REG_POOL = ['x28', 'x29', 'x30', 'x31']
PRINT_REG = 'x17'

class Allocation:
    def __init__(self):
        self.var_to_reg = {}
        self.spills = {}
        self.next_spill_offset = 0
        
    def allocate(self, vars_list):
        unique = []
        for v in vars_list:
            if v not in unique:
                unique.append(v)
                
        for i, v in enumerate(unique):
            if i < len(VAR_REG_POOL):
                self.var_to_reg[v] = VAR_REG_POOL[i]
            else:
                self.spills[v] = self.next_spill_offset
                self.next_spill_offset += 4
                
        return self.var_to_reg, self.spills

# ==========================================================
# IR LOWERING
# ==========================================================

Instr = namedtuple('Instr', ['op', 'rd', 'rs1', 'rs2', 'imm', 'comment'])

def make_instr(op, rd=None, rs1=None, rs2=None, imm=None, comment=''):
    return Instr(op, rd, rs1, rs2, imm, comment)

def lower_ast_to_ir(stmts):
    ir = []
    for s in stmts:
        if isinstance(s, Assign):
            dest = s.dest
            if isinstance(s.expr, Const):
                ir.append(('li', dest, s.expr.value))
            elif isinstance(s.expr, Var):
                ir.append(('mov', dest, s.expr.name))
            elif isinstance(s.expr, BinOp):
                left = s.expr.left
                right = s.expr.right
                L = left.value if isinstance(left, Const) else left.name
                R = right.value if isinstance(right, Const) else right.name
                ir.append(('binop', s.expr.op, dest, L, R))
        elif isinstance(s, Print):
            ir.append(('print', s.var))
    return ir

# ==========================================================
# IR -> ASSEMBLY
# ==========================================================

def ir_to_assembly(ir, alloc: Allocation):
    asm = []
    temp_idx = 0
    
    def temp_reg():
        nonlocal temp_idx
        r = TEMP_REG_POOL[temp_idx % len(TEMP_REG_POOL)]
        temp_idx += 1
        return r
    
    for ins in ir:
        kind = ins[0]
        
        if kind == 'li':
            _, dest, val = ins
            if dest in alloc.var_to_reg:
                rd = alloc.var_to_reg[dest]
                asm.append(make_instr('addi', rd, 'x0', None, val, f'{dest} = {val}'))
            else:
                t = temp_reg()
                asm.append(make_instr('addi', t, 'x0', None, val, f'{dest} = {val}'))
                asm.append(make_instr('sw', None, t, None, alloc.spills[dest], f'store {dest}'))
                
        elif kind == 'mov':
            _, dest, src = ins
            rd = alloc.var_to_reg.get(dest)
            rs = alloc.var_to_reg.get(src)
            
            if rd and rs:
                asm.append(make_instr('add', rd, rs, 'x0', None, f'{dest} = {src}'))
            elif rd and not rs:
                t = temp_reg()
                asm.append(make_instr('lw', t, None, None, alloc.spills[src], f'load {src}'))
                asm.append(make_instr('add', rd, t, 'x0', None, f'{dest} = {src}'))
            elif not rd and rs:
                asm.append(make_instr('sw', None, rs, None, alloc.spills[dest], f'store {dest}'))
            else:
                t = temp_reg()
                asm.append(make_instr('lw', t, None, None, alloc.spills[src], f'load {src}'))
                asm.append(make_instr('sw', None, t, None, alloc.spills[dest], f'store {dest}'))
                
        elif kind == 'binop':
            _, op, dest, L, R = ins
            rd = alloc.var_to_reg.get(dest)
            
            def get_operand(operand):
                if _is_int(str(operand)):
                    return ('imm', int(operand), None)
                elif operand in alloc.var_to_reg:
                    return ('reg', alloc.var_to_reg[operand], None)
                else:
                    t = temp_reg()
                    return ('mem', alloc.spills[operand], t)
            
            L_type, L_val, L_reg = get_operand(L)
            R_type, R_val, R_reg = get_operand(R)
            
            # Load operands if needed
            if L_type == 'mem':
                asm.append(make_instr('lw', L_reg, None, None, L_val, f'load {L}'))
                L_reg_final = L_reg
            elif L_type == 'imm':
                L_reg = temp_reg()
                asm.append(make_instr('addi', L_reg, 'x0', None, L_val, f'load {L_val}'))
                L_reg_final = L_reg
            else:
                L_reg_final = L_val
                
            if R_type == 'mem':
                asm.append(make_instr('lw', R_reg, None, None, R_val, f'load {R}'))
                R_reg_final = R_reg
            elif R_type == 'imm':
                R_reg = temp_reg()
                asm.append(make_instr('addi', R_reg, 'x0', None, R_val, f'load {R_val}'))
                R_reg_final = R_reg
            else:
                R_reg_final = R_val
            
            # Perform operation
            target = rd if rd else temp_reg()
            
            if op == '+':
                asm.append(make_instr('add', target, L_reg_final, R_reg_final, None, f'{dest} = {L} + {R}'))
            elif op == '-':
                asm.append(make_instr('sub', target, L_reg_final, R_reg_final, None, f'{dest} = {L} - {R}'))
            elif op == '*':
                asm.append(make_instr('mul', target, L_reg_final, R_reg_final, None, f'{dest} = {L} * {R}'))
                
            # Store result if spilled
            if not rd:
                asm.append(make_instr('sw', None, target, None, alloc.spills[dest], f'store {dest}'))
                
        elif kind == 'print':
            _, var = ins
            if var in alloc.var_to_reg:
                r = alloc.var_to_reg[var]
                asm.append(make_instr('mvprint', PRINT_REG, r, None, None, f'print {var}'))
            else:
                t = temp_reg()
                asm.append(make_instr('lw', t, None, None, alloc.spills[var], f'load {var}'))
                asm.append(make_instr('mvprint', PRINT_REG, t, None, None, f'print {var}'))
                
    return asm

# ==========================================================
# DEPENDENCY ANALYSIS & SCHEDULING
# ==========================================================

def get_reg_deps(instr: Instr):
    reads, writes = set(), set()
    op = instr.op
    
    if op in ('add', 'sub', 'mul'):
        if instr.rs1 and instr.rs1 != 'x0': reads.add(instr.rs1)
        if instr.rs2 and instr.rs2 != 'x0': reads.add(instr.rs2)
        if instr.rd: writes.add(instr.rd)
    elif op == 'addi':
        if instr.rs1 and instr.rs1 != 'x0': reads.add(instr.rs1)
        if instr.rd: writes.add(instr.rd)
    elif op == 'lw':
        if instr.rd: writes.add(instr.rd)
    elif op == 'sw':
        if instr.rs1 and instr.rs1 != 'x0': reads.add(instr.rs1)
    elif op == 'mvprint':
        if instr.rs1 and instr.rs1 != 'x0': reads.add(instr.rs1)
        if instr.rd: writes.add(instr.rd)
    
    return reads, writes

def build_dependency_graph(asm):
    n = len(asm)
    preds = {i: set() for i in range(n)}
    succs = {i: set() for i in range(n)}
    edge_lat = {}
    reads_list, writes_list = [], []
    for ins in asm:
        reads, writes = get_reg_deps(ins)
        reads_list.append(reads)
        writes_list.append(writes)
    
    for i in range(n):
        for j in range(i+1, n):
            added = False
            if writes_list[i] & reads_list[j]:
                preds[j].add(i)
                succs[i].add(j)
                edge_lat[(i, j)] = 4
                added = True
            if writes_list[i] & writes_list[j]:
                if not added:
                    preds[j].add(i)
                    succs[i].add(j)
                    added = True
                edge_lat[(i, j)] = 1
            if reads_list[i] & writes_list[j]:
                if not added:
                    preds[j].add(i)
                    succs[i].add(j)
                    added = True
                edge_lat[(i, j)] = 1
    
    return preds, succs, edge_lat

def compute_earliest_times(preds, succs, edge_lat, n):
    e = [0] * n
    in_deg = [len(preds[i]) for i in range(n)]
    ready = deque(i for i in range(n) if in_deg[i] == 0)
    while ready:
        i = ready.popleft()
        for succ in succs[i]:
            lat = edge_lat.get((i, succ), 1)
            e[succ] = max(e[succ], e[i] + lat)
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                ready.append(succ)
    return e

def list_schedule(asm):
    n = len(asm)
    if n == 0:
        return []
    
    preds, succs, edge_lat = build_dependency_graph(asm)
    
    # Classify instructions
    independent = []  # No dependencies at all
    dependent = []    # Has dependencies
    
    for i in range(n):
        if len(preds[i]) == 0 and len(succs[i]) == 0:
            independent.append(i)
        else:
            dependent.append(i)
    
    # Schedule dependent instructions in topological order
    scheduled = []
    remaining_deps = set(dependent)
    in_degree = {i: len(preds[i]) for i in dependent}
    
    # Track when each register will be ready (written to)
    reg_ready_time = {}
    independent_pool = list(independent)  # Pool of independent instructions to insert
    
    while remaining_deps or independent_pool:
        # Find ready dependent instructions
        ready_deps = [i for i in remaining_deps if in_degree[i] == 0]
        
        if not ready_deps and not independent_pool:
            break
        
        # Calculate when each ready instruction can actually execute
        can_schedule = []
        for i in ready_deps:
            reads, _ = get_reg_deps(asm[i])
            earliest = len(scheduled)
            
            # Check when all input registers are available
            for reg in reads:
                if reg in reg_ready_time:
                    # Need to wait until register is written
                    earliest = max(earliest, reg_ready_time[reg])
            
            can_schedule.append((earliest, i, True))  # (ready_time, index, is_dependent)
        
        # Add independent instructions (they can go anywhere)
        for i in independent_pool:
            can_schedule.append((len(scheduled), i, False))
        
        # Sort by ready time
        can_schedule.sort(key=lambda x: (x[0], x[1]))
        
        if not can_schedule:
            break
        
        earliest_time, selected, is_dep = can_schedule[0]
        
        # If there's a stall coming (earliest_time > len(scheduled)),
        # try to fill with independent instructions
        while earliest_time > len(scheduled) and independent_pool:
            # Insert an independent instruction to fill the bubble
            indep_instr = independent_pool.pop(0)
            scheduled.append(asm[indep_instr])
            
            # Update register timing for independent instruction
            _, writes = get_reg_deps(asm[indep_instr])
            for reg in writes:
                if reg and reg != 'x0':
                    reg_ready_time[reg] = len(scheduled) + 3  # Available after WB
        
        # Now schedule the selected instruction
        scheduled.append(asm[selected])
        
        if is_dep:
            remaining_deps.remove(selected)
            
            # Update register ready times
            _, writes = get_reg_deps(asm[selected])
            for reg in writes:
                if reg and reg != 'x0':
                    reg_ready_time[reg] = len(scheduled) + 3  # Available 4 cycles after IF (at WB)
            
            # Update successors
            for succ in succs[selected]:
                if succ in in_degree:
                    in_degree[succ] -= 1
        else:
            independent_pool.remove(selected)
    
    # Add any remaining independent instructions at the end
    for i in independent_pool:
        scheduled.append(asm[i])
    
    return scheduled if len(scheduled) == n else asm

# ==========================================================
# MACHINE CODE GENERATION
# ==========================================================

def _reg_num(r):
    if r and r.startswith('x'):
        return int(r[1:])
    return 0

def _sext_imm12(i):
    i = int(i) & 0xFFF
    if i & 0x800:
        i |= 0xFFFFF000
    return i & 0xFFF

def enc_r(rd, rs1, rs2, funct3, funct7, opcode=0x33):
    return ((funct7 & 0x7F) << 25) | (_reg_num(rs2) << 20) | (_reg_num(rs1) << 15) | \
           ((funct3 & 0x7) << 12) | (_reg_num(rd) << 7) | (opcode & 0x7F)

def enc_i(rd, rs1, imm, funct3, opcode=0x13):
    imm12 = _sext_imm12(imm)
    return (imm12 << 20) | (_reg_num(rs1) << 15) | ((funct3 & 0x7) << 12) | \
           (_reg_num(rd) << 7) | (opcode & 0x7F)

def enc_s(rs1, rs2, imm, funct3, opcode=0x23):
    imm12 = _sext_imm12(imm)
    imm_11_5 = (imm12 >> 5) & 0x7F
    imm_4_0 = imm12 & 0x1F
    return (imm_11_5 << 25) | (_reg_num(rs2) << 20) | (_reg_num(rs1) << 15) | \
           ((funct3 & 0x7) << 12) | (imm_4_0 << 7) | (opcode & 0x7F)

def encode_instr(ins: Instr):
    op = ins.op
    
    if op == 'add':
        return enc_r(ins.rd, ins.rs1, ins.rs2, 0x0, 0x00)
    elif op == 'sub':
        return enc_r(ins.rd, ins.rs1, ins.rs2, 0x0, 0x20)
    elif op == 'mul':
        return enc_r(ins.rd, ins.rs1, ins.rs2, 0x0, 0x01)
    elif op == 'addi':
        return enc_i(ins.rd, ins.rs1, ins.imm or 0, 0x0)
    elif op == 'lw':
        return enc_i(ins.rd, 'x0', ins.imm or 0, 0x2, 0x03)
    elif op == 'sw':
        return enc_s('x0', ins.rs1, ins.imm or 0, 0x2, 0x23)
    elif op == 'mvprint':
        return enc_i('x17', ins.rs1, 0, 0x0)
    else:
        return enc_i('x0', 'x0', 0, 0x0)  # NOP

def encode_program(asm_list):
    hex_lines = []
    for ins in asm_list:
        word = encode_instr(ins)
        hex_lines.append(f"0x{word:08x} ; {instr_to_text(ins)}")
    return hex_lines

def extract_machine_code(mc_lines):
    """Extract just the hex values from machine code lines"""
    instructions = []
    for line in mc_lines:
        if line.strip():
            hex_part = line.split(';')[0].strip()
            if hex_part.startswith('0x'):
                instructions.append(hex_part)
    return instructions

# ==========================================================
# PIPELINE SIMULATOR
# ==========================================================

class PipelineSimulator:
    def __init__(self, asm):
        self.asm = asm
        self.pc = 0
        self.cycle = 0
        self.pipeline = [None] * 5  # IF, ID, EX, MEM, WB
        self.timeline = []
        self.total_cycles = 0
    
    def step(self):
        self.cycle += 1
        
        # Check for RAW hazard between ID and later stages
        stall = False
        if self.pipeline[1]:  # ID stage has instruction
            id_reads, _ = get_reg_deps(self.pipeline[1])
            for stage_idx in [2, 3, 4]:  # EX, MEM, WB stages
                if self.pipeline[stage_idx]:
                    _, stage_writes = get_reg_deps(self.pipeline[stage_idx])
                    if id_reads & stage_writes:
                        stall = True
                        break
        
        # Record current pipeline state with stage names
        stage_ops = []
        for stage in self.pipeline:
            if stage:
                stage_ops.append(stage.op)
            else:
                stage_ops.append('---')
        self.timeline.append((self.cycle, stage_ops.copy()))
        
        if stall:
            # Insert bubble - don't advance ID to EX, insert NOP
            self.pipeline[4] = self.pipeline[3]
            self.pipeline[3] = self.pipeline[2] 
            self.pipeline[2] = None  # Bubble
            # Don't advance IF to ID or fetch new instruction
        else:
            # Normal advance
            for i in range(4, 0, -1):
                self.pipeline[i] = self.pipeline[i-1]
            
            # Fetch new instruction
            if self.pc < len(self.asm):
                self.pipeline[0] = self.asm[self.pc]
                self.pc += 1
            else:
                self.pipeline[0] = None
    
    def run(self, max_cycles=1000):
        while (self.pc < len(self.asm) or any(self.pipeline)) and self.cycle < max_cycles:
            self.step()
        self.total_cycles = self.cycle
        return self.cycle

# ==========================================================
# UTILITIES
# ==========================================================

def instr_to_text(ins: Instr):
    parts = [ins.op]
    if ins.rd: parts.append(ins.rd)
    if ins.rs1: parts.append(ins.rs1)
    if ins.rs2: parts.append(ins.rs2)
    if ins.imm is not None: parts.append(str(ins.imm))
    
    result = ' '.join(parts)
    if ins.comment:
        result += f"  # {ins.comment}"
    return result

# ==========================================================
# MAIN COMPILE FUNCTION
# ==========================================================

def compile_and_simulate(source):
    try:
        stmts = parse_source(source)
    except Exception as e:
        return {'error': str(e)}
    
    # Collect variables
    vars_list = []
    for s in stmts:
        if isinstance(s, Assign):
            vars_list.append(s.dest)
            def collect_vars(node):
                if isinstance(node, Var):
                    vars_list.append(node.name)
                elif isinstance(node, BinOp):
                    collect_vars(node.left)
                    collect_vars(node.right)
            collect_vars(s.expr)
        elif isinstance(s, Print):
            vars_list.append(s.var)
    
    # Allocate registers
    alloc = Allocation()
    alloc.allocate(vars_list)
    
    # Generate IR and assembly
    ir = lower_ast_to_ir(stmts)
    asm_before = ir_to_assembly(ir, alloc)
    asm_after = list_schedule(asm_before)
    
    # Generate machine code
    mc_before = encode_program(asm_before)
    mc_after = encode_program(asm_after)
    
    # Simulate both versions
    sim_before = PipelineSimulator(asm_before)
    cycles_before = sim_before.run()
    
    sim_after = PipelineSimulator(asm_after)
    cycles_after = sim_after.run()
    
    return {
        'asm_before': [instr_to_text(ins) for ins in asm_before],
        'asm_after': [instr_to_text(ins) for ins in asm_after],
        'mc_before': mc_before,
        'mc_after': mc_after,
        'cycles_before': cycles_before,
        'cycles_after': cycles_after,
        'timeline_before': sim_before.timeline,
        'timeline_after': sim_after.timeline,
        'var_alloc': alloc.var_to_reg,
        'spills': alloc.spills
    }

def run_emulator_with_machine_code(machine_code):
    """Run the emulator with given machine code and return register state"""
    try:
        instructions = extract_machine_code(machine_code)
        cpu = SimpleRV32I(instructions=instructions)
        reg_state, write_log = cpu.run(max_steps=200, rst=1)
        return {
            'register_state': reg_state,
            'write_log': write_log,
            'error': None
        }
    except Exception as e:
        return {
            'register_state': {},
            'write_log': [],
            'error': str(e)
        }

# ==========================================================
# GUI APPLICATION
# ==========================================================

EXAMPLE = '''
b = 5
c = 6  
a = b + c
d = a * 2
'''

class App:
    def __init__(self, root):
        self.root = root
        root.title('RISC-V Static Scheduling IDE with Emulator')
        root.geometry('1400x900')
        self.setup_widgets()
        
    def setup_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Source code input
        ttk.Label(main_frame, text="Source (tiny C-like subset):").grid(row=0, column=0, columnspan=2, sticky="w")
        
        source_frame = ttk.Frame(main_frame)
        source_frame.grid(row=1, column=0, sticky="nsew", pady=(5,0), padx=(0,5))
        source_frame.columnconfigure(0, weight=1)
        source_frame.rowconfigure(0, weight=1)
        
        self.source_text = scrolledtext.ScrolledText(source_frame, width=60, height=8)
        self.source_text.grid(row=0, column=0, sticky="nsew")
        self.source_text.insert('1.0', EXAMPLE)
        
        # Compile and Run buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
        
        ttk.Button(button_frame, text="Compile & Analyze", command=self.compile_run).pack(side="left", padx=(0,10))
        ttk.Button(button_frame, text="Run Emulator", command=self.run_emulator).pack(side="left")
        self.cycle_label = ttk.Label(button_frame, text="")
        self.cycle_label.pack(side="left", padx=(20,0))
        
        # Results tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10,0))
        main_frame.rowconfigure(3, weight=2)
        
        # Create tabs
        self.tab_before = ttk.Frame(self.notebook)
        self.tab_after = ttk.Frame(self.notebook)
        self.tab_mc_before = ttk.Frame(self.notebook)
        self.tab_mc_after = ttk.Frame(self.notebook)
        self.tab_emulator = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_before, text="Timeline - before")
        self.notebook.add(self.tab_after, text="Timeline - after")
        self.notebook.add(self.tab_mc_before, text="Machine Code (before)")
        self.notebook.add(self.tab_mc_after, text="Machine Code (after)")
        self.notebook.add(self.tab_emulator, text="Emulator Results")
        
        # Setup tabs
        self.setup_timeline_tab(self.tab_before, "before")
        self.setup_timeline_tab(self.tab_after, "after")
        self.setup_machine_code_tab(self.tab_mc_before, "before")
        self.setup_machine_code_tab(self.tab_mc_after, "after")
        self.setup_emulator_tab(self.tab_emulator)
        
        # Register allocation display (right panel)
        alloc_frame = ttk.LabelFrame(main_frame, text="Register Allocation:")
        alloc_frame.grid(row=1, column=1, sticky="nsew", pady=(5,0))
        alloc_frame.columnconfigure(0, weight=1)
        alloc_frame.rowconfigure(0, weight=1)
        
        self.reg_text = scrolledtext.ScrolledText(alloc_frame, width=25, height=20)
        self.reg_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Timeline buttons
        timeline_frame = ttk.Frame(main_frame)
        timeline_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")
        
        ttk.Button(timeline_frame, text="Show Timeline (before)", 
                  command=lambda: self.show_timeline_popup("before")).pack(side="left", padx=(0,10))
        ttk.Button(timeline_frame, text="Show Timeline (after)", 
                  command=lambda: self.show_timeline_popup("after")).pack(side="left")
        
    def setup_timeline_tab(self, parent, which):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        text_widget = scrolledtext.ScrolledText(parent, font=('Courier', 10))
        text_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        stats_label = ttk.Label(parent, text="")
        stats_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        if which == "before":
            self.asm_before_text = text_widget
            self.stats_before_label = stats_label
        else:
            self.asm_after_text = text_widget
            self.stats_after_label = stats_label
    
    def setup_machine_code_tab(self, parent, which):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        text_widget = scrolledtext.ScrolledText(parent, font=('Courier', 9))
        text_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        if which == "before":
            self.mc_before_text = text_widget
        else:
            self.mc_after_text = text_widget
    
    def setup_emulator_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # Create notebook for emulator results
        emulator_notebook = ttk.Notebook(parent)
        emulator_notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Register state tab
        reg_state_frame = ttk.Frame(emulator_notebook)
        emulator_notebook.add(reg_state_frame, text="Register State")

        reg_state_frame.columnconfigure(0, weight=1)
        reg_state_frame.rowconfigure(0, weight=1)

        self.emulator_reg_text = scrolledtext.ScrolledText(reg_state_frame, font=('Courier', 10), width=60, height=20)
        self.emulator_reg_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Execution log tab
        exec_log_frame = ttk.Frame(emulator_notebook)
        emulator_notebook.add(exec_log_frame, text="Execution Log")

        exec_log_frame.columnconfigure(0, weight=1)
        exec_log_frame.rowconfigure(0, weight=1)

        self.emulator_log_text = scrolledtext.ScrolledText(exec_log_frame, font=('Courier', 9), width=60, height=20)
        self.emulator_log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Stack memory tab
        stack_frame = ttk.Frame(emulator_notebook)
        emulator_notebook.add(stack_frame, text="Stack Memory")

        stack_frame.columnconfigure(0, weight=1)
        stack_frame.rowconfigure(0, weight=1)

        self.emulator_stack_text = scrolledtext.ScrolledText(stack_frame, font=('Courier', 10), width=50, height=20)
        self.emulator_stack_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.emulator_notebook = emulator_notebook

    def compile_run(self):
        source = self.source_text.get('1.0', 'end-1c')
        result = compile_and_simulate(source)
        
        if 'error' in result:
            messagebox.showerror("Compilation Error", result['error'])
            return
        
        # Update assembly displays
        self.asm_before_text.delete('1.0', 'end')
        self.asm_before_text.insert('1.0', '\n'.join(result['asm_before']))
        
        self.asm_after_text.delete('1.0', 'end')
        self.asm_after_text.insert('1.0', '\n'.join(result['asm_after']))
        
        # Update machine code displays
        self.mc_before_text.delete('1.0', 'end')
        self.mc_before_text.insert('1.0', '\n'.join(result['mc_before']))
        
        self.mc_after_text.delete('1.0', 'end')
        self.mc_after_text.insert('1.0', '\n'.join(result['mc_after']))
        
        # Update stats
        self.stats_before_label.config(text=f"Cycles: {result['cycles_before']}")
        self.stats_after_label.config(text=f"Cycles: {result['cycles_after']}")
        self.cycle_label.config(text=f"Cycles: {result['cycles_after']}")
        
        # Update register allocation
        self.reg_text.delete('1.0', 'end')
        if result['var_alloc']:
            for var, reg in result['var_alloc'].items():
                self.reg_text.insert('end', f"{var} -> {reg}\n")
        if result['spills']:
            self.reg_text.insert('end', "\nSpilled:\n")
            for var, offset in result['spills'].items():
                self.reg_text.insert('end', f"{var} -> stack[{offset}]\n")
        
        # Store results for timeline display and emulator
        self.last_result = result
        
        # Clear emulator results
        self.emulator_reg_text.delete('1.0', 'end')
        self.emulator_reg_text.insert('1.0', 'Click "Run Emulator" to execute the optimized machine code')
        
        self.emulator_log_text.delete('1.0', 'end')
        self.emulator_log_text.insert('1.0', 'Click "Run Emulator" to see execution log')
    
    def run_emulator(self):
        if not hasattr(self, 'last_result'):
            messagebox.showinfo("Info", "Please compile first")
            return

        machine_code = self.last_result['mc_after']
        emulator_result = run_emulator_with_machine_code(machine_code)

        if emulator_result['error']:
            messagebox.showerror("Emulator Error", emulator_result['error'])
            return

        # === Register state display ===
        self.emulator_reg_text.delete('1.0', 'end')
        if emulator_result['register_state']:
            self.emulator_reg_text.insert('end', "=== Final Register State ===\n")
            for reg, info in emulator_result['register_state'].items():
                self.emulator_reg_text.insert('end', f"{reg} = {info['hex']} ({info['decimal']})\n")
        else:
            self.emulator_reg_text.insert('end', "All registers are zero (0x00000000)\n")

        # === Execution log display ===
        self.emulator_log_text.delete('1.0', 'end')
        if emulator_result['write_log']:
            self.emulator_log_text.insert('end', "=== Execution Log ===\n")
            for log_entry in emulator_result['write_log']:
                self.emulator_log_text.insert('end', f"{log_entry}\n")
        else:
            self.emulator_log_text.insert('end', "No register writes occurred during execution\n")

        # === Stack memory display ===
        self.emulator_stack_text.delete('1.0', 'end')
        try:
            # Convert machine code text to list of ints
            machine_code_lines = [line.strip() for line in machine_code if line.strip()]
            instructions = [int(line.split(';')[0], 16) for line in machine_code_lines]

            # Run emulator again to inspect data memory
            cpu = SimpleRV32I(instructions=instructions)
            cpu.run(max_steps=200, rst=1)

            # Header
            self.emulator_stack_text.insert('end', "=== Stack Memory (Spilled Variables) ===\n")

            # Display spilled variables and their current values
            spills = self.last_result.get('spills', {})
            if spills:
                for var, offset in spills.items():
                    word_index = (int(offset) >> 2) & 0x3FF
                    val = cpu.dmem.mem[word_index]
                    self.emulator_stack_text.insert(
                        'end',
                        f"{var:8s} -> stack[{offset:3d}] = 0x{val:08X} ({s32(val)})\n"
                    )
            else:
                self.emulator_stack_text.insert('end', "No spilled variables found.\n")

            # Dynamically show all stack addresses used
            self.emulator_stack_text.insert('end', "\n--- Raw Stack Memory (All Used) ---\n")
            if spills:
                max_offset = max(spills.values())
                max_index = (max_offset // 4) + 1
            else:
                max_index = 32

            for i in range(max_index):
                val = cpu.dmem.mem[i]
                if val != 0:
                    self.emulator_stack_text.insert('end', f"[{i*4:03d}] = 0x{val:08X} ({s32(val)})\n")

        except Exception as e:
            self.emulator_stack_text.insert('end', f"Error reading stack: {e}\n")

    def show_timeline_popup(self, which):
        if not hasattr(self, 'last_result'):
            messagebox.showinfo("Info", "Please compile first")
            return
        
        timeline = self.last_result[f'timeline_{which}']
        
        popup = tk.Toplevel(self.root)
        popup.title(f"Timeline - {which}")
        popup.geometry("700x500")
        
        text_widget = scrolledtext.ScrolledText(popup, font=('Courier', 10))
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        text_widget.insert('end', "Cycle |  IF   |  ID   |  EX   | MEM   |  WB  \n")
        text_widget.insert('end', "=" * 50 + "\n")
        
        for cycle, stages in timeline:
            padded_stages = stages + ['---'] * (5 - len(stages)) if len(stages) < 5 else stages
            line = f"{cycle:5d} | {padded_stages[0]:5s} | {padded_stages[1]:5s} | {padded_stages[2]:5s} | {padded_stages[3]:5s} | {padded_stages[4]:5s}\n"
            text_widget.insert('end', line)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
