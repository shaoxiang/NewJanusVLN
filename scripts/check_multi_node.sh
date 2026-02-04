#!/bin/bash
#================================================================
# Multi-Node Training Configuration Checker
#================================================================
# This script checks if your environment is ready for multi-node
# training with InfiniBand and provides diagnostic information.
#
# Usage:
#   bash scripts/check_multi_node.sh
#================================================================

set -e

echo "============================================"
echo "Multi-Node Training Configuration Check"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

# 1. Check Python and PyTorch
echo "[1] Python and PyTorch Environment"
echo "-----------------------------------"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    check_pass "Python found: $PYTHON_VERSION"
    
    # Check PyTorch
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT_FOUND")
    if [[ "$TORCH_VERSION" != "NOT_FOUND" ]]; then
        check_pass "PyTorch found: $TORCH_VERSION"
        
        # Check CUDA
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
        if [[ "$CUDA_AVAILABLE" == "True" ]]; then
            CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            check_pass "CUDA available: $CUDA_VERSION ($GPU_COUNT GPUs)"
        else
            check_fail "CUDA not available"
        fi
    else
        check_fail "PyTorch not found"
    fi
else
    check_fail "Python not found"
fi
echo ""

# 2. Check DeepSpeed
echo "[2] DeepSpeed"
echo "-----------------------------------"
DEEPSPEED_VERSION=$(python -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null || echo "NOT_FOUND")
if [[ "$DEEPSPEED_VERSION" != "NOT_FOUND" ]]; then
    check_pass "DeepSpeed found: $DEEPSPEED_VERSION"
else
    check_fail "DeepSpeed not found (pip install deepspeed)"
fi
echo ""

# 3. Check InfiniBand
echo "[3] InfiniBand Configuration"
echo "-----------------------------------"
if command -v ibv_devices &> /dev/null; then
    IB_DEVICES=$(ibv_devices 2>/dev/null | grep -v "^$")
    if [[ -n "$IB_DEVICES" ]]; then
        check_pass "InfiniBand devices found:"
        echo "$IB_DEVICES" | sed 's/^/    /'
        
        # Check IB status
        if command -v ibstat &> /dev/null; then
            IB_STATE=$(ibstat 2>/dev/null | grep "State:" | head -1)
            if echo "$IB_STATE" | grep -q "Active"; then
                check_pass "InfiniBand is Active"
            else
                check_warn "InfiniBand state: $IB_STATE"
            fi
        fi
        
        # Check IPoIB interfaces (optional)
        IB_IFACES=$(ifconfig 2>/dev/null | grep -o "^ib[0-9]*" || ip addr 2>/dev/null | grep -o "\bib[0-9]*\b" | sort -u | tr '\n' ' ' | sed 's/ $//' || echo "")
        if [[ -n "$IB_IFACES" ]]; then
            check_pass "IPoIB interfaces found: $IB_IFACES"
        else
            check_warn "No IPoIB interface (ib0/ib1) found. This is OK if you use IB verbs/RDMA without IPoIB."
            if command -v ip &> /dev/null && [[ -n "${MASTER_ADDR:-}" ]]; then
                ROUTE_DEV=$(ip -o route get "${MASTER_ADDR}" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}' || true)
                if [[ -n "$ROUTE_DEV" ]]; then
                    echo "    Suggest: export NCCL_SOCKET_IFNAME=$ROUTE_DEV"
                fi
            fi
        fi
    else
        check_warn "No InfiniBand devices found (will use Ethernet)"
        echo "    Set NCCL_IB_DISABLE=1 if using Ethernet only"
    fi
else
    check_warn "ibv_devices command not found (libibverbs not installed?)"
    echo "    If using Ethernet, set NCCL_IB_DISABLE=1"
fi
echo ""

# 4. Check NCCL
echo "[4] NCCL Configuration"
echo "-----------------------------------"
NCCL_VERSION=$(python -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null || echo "NOT_FOUND")
if [[ "$NCCL_VERSION" != "NOT_FOUND" ]]; then
    check_pass "NCCL version: $NCCL_VERSION"
else
    check_warn "NCCL version not detected"
fi

# Check current NCCL environment variables
if [[ -n "$NCCL_IB_DISABLE" ]]; then
    if [[ "$NCCL_IB_DISABLE" == "0" ]]; then
        check_pass "NCCL_IB_DISABLE=0 (InfiniBand enabled)"
    else
        check_warn "NCCL_IB_DISABLE=1 (InfiniBand disabled, using Ethernet)"
    fi
else
    check_warn "NCCL_IB_DISABLE not set (will use default)"
fi

echo ""

# 5. Check Network Connectivity (if multi-node)
echo "[5] Network Configuration"
echo "-----------------------------------"
HOSTNAME=$(hostname)
check_pass "Hostname: $HOSTNAME"

if [[ -n "$MASTER_ADDR" ]]; then
    check_pass "MASTER_ADDR set: $MASTER_ADDR"
    MASTER_PORT=${MASTER_PORT:-29500}
    
    # Try to ping master
    if ping -c 1 -W 2 "$MASTER_ADDR" &> /dev/null; then
        check_pass "Can ping MASTER_ADDR ($MASTER_ADDR)"
    else
        check_fail "Cannot ping MASTER_ADDR ($MASTER_ADDR)"
    fi
    
    # Check if port is accessible (if this is not the master node)
    if [[ "${NODE_RANK:-0}" != "0" ]]; then
        if timeout 2 bash -c "cat < /dev/null > /dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null; then
            check_pass "Master port $MASTER_PORT is accessible"
        else
            check_warn "Cannot connect to $MASTER_ADDR:$MASTER_PORT (master may not be running yet)"
        fi
    fi
else
    check_warn "MASTER_ADDR not set (single-node mode)"
fi
echo ""

# 6. Check Training Script
echo "[6] Training Script"
echo "-----------------------------------"
TRAIN_SCRIPT="scripts/train_2node_h800.sh"
if [[ -f "$TRAIN_SCRIPT" ]]; then
    check_pass "Training script found: $TRAIN_SCRIPT"
    if [[ -x "$TRAIN_SCRIPT" ]]; then
        check_pass "Training script is executable"
    else
        check_warn "Training script is not executable (run: chmod +x $TRAIN_SCRIPT)"
    fi
else
    check_fail "Training script not found: $TRAIN_SCRIPT"
fi

DS_CONFIG="${DS_CONFIG:-scripts/zero3.json}"
if [[ -f "$DS_CONFIG" ]]; then
    check_pass "DeepSpeed config found: $DS_CONFIG"
else
    check_fail "DeepSpeed config not found: $DS_CONFIG"
    check_warn "Try: export DS_CONFIG=scripts/zero2.json (or scripts/zero3.json/scripts/zero2_offload.json)"
fi
echo ""

# 7. Check Paths (if environment variables are set)
echo "[7] Data and Model Paths"
echo "-----------------------------------"
if [[ -n "$MODEL_PATH" ]]; then
    if [[ -d "$MODEL_PATH" ]]; then
        check_pass "MODEL_PATH exists: $MODEL_PATH"
    else
        check_fail "MODEL_PATH not found: $MODEL_PATH"
    fi
else
    check_warn "MODEL_PATH not set (will use default from script)"
fi

if [[ -n "$VGGT_MODEL_PATH" ]]; then
    if [[ -d "$VGGT_MODEL_PATH" ]]; then
        check_pass "VGGT_MODEL_PATH exists: $VGGT_MODEL_PATH"
    else
        check_fail "VGGT_MODEL_PATH not found: $VGGT_MODEL_PATH"
    fi
else
    check_warn "VGGT_MODEL_PATH not set (will use default from script)"
fi

if [[ -n "$DATA_ROOT" ]]; then
    if [[ -d "$DATA_ROOT" ]]; then
        check_pass "DATA_ROOT exists: $DATA_ROOT"
        # Count episodes
        EPISODE_COUNT=$(find "$DATA_ROOT" -type f -name "milestones_result.json" 2>/dev/null | wc -l)
        if [[ $EPISODE_COUNT -gt 0 ]]; then
            check_pass "Found $EPISODE_COUNT episodes"
        else
            check_warn "No episodes found (no milestones_result.json files)"
        fi
    else
        check_fail "DATA_ROOT not found: $DATA_ROOT"
    fi
else
    check_warn "DATA_ROOT not set (will use default from script)"
fi
echo ""

# 8. Summary and Recommendations
echo "============================================"
echo "Summary and Recommendations"
echo "============================================"
echo ""

if [[ -n "$IB_DEVICES" ]]; then
    echo "✓ Your system has RDMA-capable devices (InfiniBand/RoCE). Recommended configuration:"
    echo "  export NCCL_IB_DISABLE=0"
    echo "  export NCCL_IB_HCA=mlx5  # or a comma list like mlx5_0,mlx5_1,... (exclude RoCE ports if needed)"
    echo "  # NCCL_SOCKET_IFNAME should be an interface that can reach MASTER_ADDR (may be eth/bond even if using IB verbs)"
    echo ""
else
    echo "⚠ No RDMA device detected. For Ethernet-only:"
    echo "  export NCCL_IB_DISABLE=1"
    echo "  export NCCL_SOCKET_IFNAME=eth0  # or your ethernet interface"
    echo ""
fi

echo "For multi-node training:"
echo "  Node 0 (Master):"
echo "    export MASTER_ADDR=<master_ip>"
echo "    export NODE_RANK=0"
echo "    export NNODES=2"
echo "    bash scripts/train_2node_h800.sh"
echo ""
echo "  Node 1 (Worker):"
echo "    export MASTER_ADDR=<master_ip>"
echo "    export NODE_RANK=1"
echo "    export NNODES=2"
echo "    bash scripts/train_2node_h800.sh"
echo ""

echo "For more details, see: TWO_NODE_TRAINING_GUIDE.md"
echo ""
echo "============================================"
echo "Check completed at $(date)"
echo "============================================"
