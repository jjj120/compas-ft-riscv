/*
 * Copyright 2021 Chair of EDA, Technical University of Munich
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "RISCVRasm.h"

#include "RISCVSubtarget.h"
#include "common.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

llvm::FunctionPass *llvm::createRISCVRasm() { return new RISCVRasm(); }

RISCVRasm::RISCVRasm() : RISCVDmr{} {}

bool RISCVRasm::runOnMachineFunction(llvm::MachineFunction &MF) {
  MF_ = &MF;
  TII_ = MF_->getSubtarget().getInstrInfo();

  if (!riscv_common::inCSString(llvm::cl::enable_rasm, std::string{MF_->getName()})) {
    return false;
  }

  llvm::outs() << "Running RASM pass on " << MF_->getName() << "\n";

  init();
  harden();

  return true;
}

void RISCVRasm::init() {
  mbb_sigs_.clear();
  err_bbs_.clear();

  config_.eds = riscv_common::ErrorDetectionStrategy::ED0; // default value

  llvm::outs() << "COMPAS: RASM: Error Detection Strategy: " << static_cast<int>(llvm::cl::error_detection_strat)
               << "\n";

  config_.eds = static_cast<riscv_common::ErrorDetectionStrategy>((int)llvm::cl::error_detection_strat);

  // ED0: on error-detection, jump to error-block and keep on executing it
  // ED1: same as ED0 but it quits after notifying safety-unit
  // ED2: on error-detection, call a function passing the specific checker's code
  // ED3: on error-detection, jump to error block and immediately stop execution
  // ED4: on error-detection, send a pc to uart
  switch (config_.eds) {
  case riscv_common::ErrorDetectionStrategy::ED0:
    // insert an error-BB in MF_
    // insertErrorBB(std::string("RASM_UNKNOWN_ERR_BB_NAME"), -1); // probably not needed, since we want to generate
    // new error-BBs for each check
    break;
  case riscv_common::ErrorDetectionStrategy::ED1:
    // insert an error-BB in MF_
    // insertErrorBB(std::string("RASM_UNKNOWN_ERR_BB_NAME"), -1); // probably not needed, since we want to generate
    // new error-BBs for each check
    break;
  case riscv_common::ErrorDetectionStrategy::ED2:
    // assert(0 && "TODO: ED2 not implemented yet");
    break;
  case riscv_common::ErrorDetectionStrategy::ED3:
    // nothing to initialize here
    assert(0 && "TODO: ED3 not implemented yet");
    break;
  case riscv_common::ErrorDetectionStrategy::ED4:
    assert(0 && "TODO: ED4 not implemented yet");
    break;

  default:
    assert(0 && "Unknown error-detection strategy");
    break;
  }

  // assigning compile time sigs to each MBB
  std::set<short> sigs_sofar{}, sum_sofar{};
  for (auto &MBB : *MF_) {
    // there could be err-BBs from other passes
    // the convention is if a BB keeps jumping to itself then this is also
    // an error-BB
    if (MBB.succ_size() == 1 && &MBB == *MBB.succ_begin()) {
      err_bbs_.emplace(&MBB);
      continue;
    }

    while (1) {
      auto x{unif_dist_(gen_)};
      if (sigs_sofar.find(x) == std::end(sigs_sofar)) {
        sigs_sofar.emplace(x);
        while (1) {
          auto y{unif_dist_(gen_)};
          if (sum_sofar.find(x + y) == std::end(sum_sofar)) {
            sum_sofar.emplace(x + y);
            mbb_sigs_[&MBB] = {x, y};
            break;
          }
        }
        break;
      }
    }
  }
}

// RASM hardening scheme
void RISCVRasm::harden() {
  for (auto &MBB : *MF_) {
    // there could be err-BBs from other passes
    // the convention is if a BB keeps jumping to itself then this is also
    // an error-BB
    if (MBB.succ_size() == 1 && &MBB == *MBB.succ_begin()) {
      continue;
    }

    if (MBB.pred_empty()) {
      // for entryBB we dont check RTS
      llvm::BuildMI(MF_->front(), std::begin(MF_->front()), std::begin(MF_->front())->getDebugLoc(),
                    TII_->get(llvm::RISCV::ADDI))
          .addReg(kRTS)
          .addReg(riscv_common::k0)
          .addImm(mbb_sigs_[&MF_->front()].first);
    } else {
      auto insert{std::begin(MBB)};
      // updating RTS
      llvm::BuildMI(MBB, insert, insert->getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
          .addReg(kRTS)
          .addReg(kRTS)
          .addImm(-mbb_sigs_[&MBB].second);
      // comparing the RTS with compile-time sig
      llvm::BuildMI(MBB, insert, insert->getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
          .addReg(kC)
          .addReg(riscv_common::k0)
          .addImm(mbb_sigs_[&MBB].first);

      llvm::BuildMI(MBB, insert, insert->getDebugLoc(), TII_->get(llvm::RISCV::BNE))
          .addReg(kRTS)
          .addReg(kC)
          .addMBB(insertErrorBB(std::string("RASM_ERR_BB_UNKNOWN_1"), -1));
    }

    // special processing for some individual instrs:
    // we generate new instrs in the following loop which should not be
    // considered in future iterations. For this we use following flag
    bool ignore_this_mi{false};
    // in case we have a conditional and unconditional branch both in MBB then
    // we dont need to update signature before the final unconditional branch
    // we use the following flag to keep track of this
    bool ignore_jump{false};
    for (auto &MI : MBB) {
      if (ignore_this_mi) {
        ignore_this_mi = false;
        continue;
      }

      // in case we call another function then RTS would be corrupted
      // as we use t0 RISCV register for RTS, hence
      // preserving them across function call using the stack
      if (MI.isCall()) {
        // jumps are also considered as calls so filtering them out
        if (MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == llvm::RISCV::X0) {
          continue;
        }
        // filtering out tail calls
        if (TII_->isTailCall(MI)) {
          continue;
        }

        llvm::MachineBasicBlock::iterator insert{MI.getIterator()};
        insert++;

        // NOTE: stack would be used to pass args if they exceed arg regs hence
        //       we shouldn't use stack for CFC signatures

        if (MI.getOpcode() == llvm::RISCV::PseudoCALLIndirect ||
            !(riscv_common::inCSString(llvm::cl::enable_nzdc, std::string{MF_->getName()}) &&
              riscv_common::setmapContains(knownLibcalls2Duplicable_, getCalledFuncName(&MI)))) {
          // if we are in DMR func and we are going to libcall via this MI call
          // then no need to stack RTS explicitly as this is already done in DMR
          // else: we stack RTS as below

          riscv_common::saveRegs({kRTS}, &MBB, MI.getIterator());
          riscv_common::loadRegs({kRTS}, &MBB, insert);

          if (riscv_common::inCSString(llvm::cl::enable_nzdc, std::string{MF_->getName()})) {
            llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
                .addReg(P2S_.at(riscv_common::kSP))
                .addReg(P2S_.at(riscv_common::kSP))
                .addImm(-4);
            llvm::BuildMI(MBB, insert, MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
                .addReg(P2S_.at(riscv_common::kSP))
                .addReg(P2S_.at(riscv_common::kSP))
                .addImm(4);
          }
        }

        // alternatively: for G, we simply move the expected value back into G
        //        llvm::BuildMI(MBB, insert, MI.getDebugLoc(),
        //                      TII_->get(llvm::RISCV::ADDI))
        //            .addReg(kRTS)
        //            .addReg(riscv_common::k0)
        //            .addImm(mbb_sigs_[&MBB].first);

        continue;
      }

      // RASM processing around the branch terminators
      if (MI.isConditionalBranch()) {
        assert(MI.getOperand(2).isMBB());
        auto SBB{MI.getOperand(2).getMBB()};
        // filtering error-handler jumps
        if (err_bbs_.find(SBB) != err_bbs_.end()) {
          // this could be the loadback check at the last
          // if thats the case then have to update RASM update
          if (std::find(err_bb_vec_.begin(), err_bb_vec_.end(), SBB) != err_bb_vec_.end() && &MBB.back() == &MI) {
            for (auto &op : MI.operands()) {
              if (op.isReg() && P2S_.find(op.getReg()) != P2S_.end()) {
                assert(MBB.succ_size() == 1);
                auto SBB{*std::begin(MBB.successors())};
                assert(std::abs(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first) < 2045 &&
                       "large sig generated\n");
                llvm::BuildMI(MBB, std::end(MBB), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
                    .addReg(kRTS)
                    .addReg(kRTS)
                    .addImm(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first);
                ignore_this_mi = true;

                break;
              }
            }
          }

          continue;
        }

        assert(std::abs(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first) < 2045 &&
               "large sig generated\n");
        llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
            .addReg(kRTS)
            .addReg(kRTS)
            .addImm(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first);

        llvm::MachineBasicBlock *other_SBB{nullptr};
        for (auto SBBx : MBB.successors()) {
          if (SBBx != SBB) {
            other_SBB = SBBx;
            break;
          }
        }

        assert(std::abs(-(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first) +
                        (mbb_sigs_[other_SBB].first + mbb_sigs_[other_SBB].second - mbb_sigs_[&MBB].first)) < 2045 &&
               "large sig generated");
        llvm::BuildMI(MBB, std::next(MI.getIterator()), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
            .addReg(kRTS)
            .addReg(kRTS)
            .addImm(-(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first) +
                    (mbb_sigs_[other_SBB].first + mbb_sigs_[other_SBB].second - mbb_sigs_[&MBB].first));
        ignore_this_mi = true;
        if (MBB.back().isUnconditionalBranch()) {
          ignore_jump = true;
        } else {
          if (MBB.back().isCall() && MBB.back().getOperand(1).isMBB() &&
              err_bbs_.find(MBB.back().getOperand(1).getMBB()) != err_bbs_.end()) {
            ignore_jump = true;
          }
        }

        continue;
      }
      if (MI.isUnconditionalBranch()) {
        if (!ignore_jump) {
          assert(MI.getOperand(0).isMBB());
          auto SBB{MI.getOperand(0).getMBB()};

          assert(std::abs(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first) < 2045 &&
                 "large sig generated\n");
          llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
              .addReg(kRTS)
              .addReg(kRTS)
              .addImm(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first);
        }

        continue;
      }

      // RASM processing before the return instruction
      if (MI.isReturn()) {
        short rand{unif_dist_(gen_)};
        llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
            .addReg(kRTS)
            .addReg(kRTS)
            .addImm(rand - mbb_sigs_[&MBB].first);
        llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
            .addReg(kC)
            .addReg(riscv_common::k0)
            .addImm(rand);

        llvm::BuildMI(MBB, MI.getIterator(), MI.getDebugLoc(), TII_->get(llvm::RISCV::BNE))
            .addReg(kRTS)
            .addReg(kC)
            .addMBB(insertErrorBB(std::string("RASM_ERR_BB_UNKNOWN"), -1));
        break;
      }

      // in case this MBB has no branch/ret terminator then we have to do RASM
      // update at end of MBB
      if (&MI == &MBB.back() && MBB.succ_size()) {
        assert(MBB.succ_size() == 1);
        auto SBB{*std::begin(MBB.successors())};

        assert(std::abs(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first) < 2045 &&
               "large sig generated\n");
        llvm::BuildMI(MBB, std::end(MBB), MI.getDebugLoc(), TII_->get(llvm::RISCV::ADDI))
            .addReg(kRTS)
            .addReg(kRTS)
            .addImm(mbb_sigs_[SBB].first + mbb_sigs_[SBB].second - mbb_sigs_[&MBB].first);
        break;
      }
    }
  }
}

// for now we just write '1' to special memory address (0xfff8) to tell the
// simulator that this FI is covered in a separate error-BB. Further we
// stay in this loop to avoid further functional execution
llvm::MachineBasicBlock *RISCVRasm::insertErrorBB(std::string name, int counter) {
  llvm::MachineBasicBlock *cf_err_bb_ = MF_->CreateMachineBasicBlock();
  err_bb_vec_.push_back(cf_err_bb_);
  MF_->push_back(cf_err_bb_);

  auto DLL{MF_->front().front().getDebugLoc()};

  // ED0: on error-detection, jump to error-block and keep on executing it (0xFFF8)
  // ED1: same as ED0 but it quits after notifying safety-unit (0xFFF8)
  // ED2: on error-detection, call a function passing the specific checker's code
  // ED3: on error-detection, jump to error block and immediately stop execution
  // ED4: on error-detection, send pc to uart
  switch (config_.eds) // error-detection strategy
  {
  case riscv_common::ErrorDetectionStrategy::ED0:
    // storing '1' to addr : (0xfff8 = 0x10000 - 0x8):
    // lui t1, 16 -> makes t1 = 0x10000
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::LUI))
        .addReg(llvm::RISCV::X6)
        .addImm(16);
    // addi t2, zero, 1 -> makes t2 = 1
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::ADDI))
        .addReg(llvm::RISCV::X7)
        .addReg(riscv_common::k0)
        .addImm(1);
    // sw t2, -8(t1) -> stores t2 to (t1 - 8) i.e. store 1 to 0xfff8
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(isa_config_.store_opcode))
        .addReg(llvm::RISCV::X7)
        .addReg(llvm::RISCV::X6)
        .addImm(-8);

    // keep on repeating this errBB as we dont want to execute code now
    // J cf_err_bb_ = JALR X0, cf_err_bb_ because J is a pseudo-jump instr in
    // RISCV
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::JAL))
        .addReg(riscv_common::k0)
        .addMBB(cf_err_bb_);
    break;

  case riscv_common::ErrorDetectionStrategy::ED1:
    // storing '1' to addr : (0xfff8 = 0x10000 - 0x8):
    // lui t1, 16 -> makes t1 = 0x10000
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::LUI))
        .addReg(llvm::RISCV::X6)
        .addImm(16);
    // addi t2, zero, 1 -> makes t2 = 1
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::ADDI))
        .addReg(llvm::RISCV::X7)
        .addReg(riscv_common::k0)
        .addImm(1);
    // sw t2, -8(t1) -> stores t2 to (t1 - 8) i.e. store 1 to 0xfff8
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(isa_config_.store_opcode))
        .addReg(llvm::RISCV::X7)
        .addReg(llvm::RISCV::X6)
        .addImm(-8);

    // quit immediately after notifying safety-unit
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::EBREAK));
    break;

  case riscv_common::ErrorDetectionStrategy::ED2:
    // assert(0 && "ED2 should not use error-BB");
    break;

  case riscv_common::ErrorDetectionStrategy::ED3:
    llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::EBREAK));
    break;

  case riscv_common::ErrorDetectionStrategy::ED4:
    assert(0 && "ED4 not implemented yet");
    break;

  default:
    assert(0 && "Unknown error-detection strategy");
    break;
  }

  // llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::ANNOTATION_LABEL))
  //     .addImm(0); // 0 is for no annotation type
  // .addExternalSymbol(name.c_str(), llvm::MCSymbolRefExpr::VK_None)
  // .addImm(counter)

  // llvm::BuildMI(*cf_err_bb_, std::end(*cf_err_bb_), DLL, TII_->get(llvm::RISCV::DBG_LABEL))
  //     .addExternalSymbol(("dbg: " + name).c_str(), llvm::MCSymbolRefExpr::VK_None);

  // to encode the error-BB in order to make it a label so as to be able to
  // jump to it from anywhere in asm
  cf_err_bb_->addSuccessor(cf_err_bb_);

  llvm::outs() << "RASM: created error-BB " << cf_err_bb_->getFullName() << " for check " << name << "\n";


  return cf_err_bb_;
}
