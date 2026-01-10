import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class DynamicLatentMemoryCompressor(nn.Module):
    """
    动态潜在记忆压缩器
    核心思想：将长对话历史压缩为固定大小的连续潜在记忆向量
    """

    def __init__(
        self,
        llm_hidden_size: int = 4096,  # LLM隐藏层维度
        memory_dim: int = 512,  # 压缩后的记忆维度
        num_memory_slots: int = 16,  # 记忆槽数量
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.num_memory_slots = num_memory_slots

        # 1. 记忆编码器：将LLM隐藏状态压缩到记忆空间
        self.memory_encoder = nn.Sequential(
            nn.Linear(llm_hidden_size, memory_dim * 2),
            nn.LayerNorm(memory_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim * 2, memory_dim),
        )

        # 2. 可学习的记忆槽（类似于Perceiver的潜在变量）
        self.memory_slots = nn.Parameter(torch.randn(num_memory_slots, memory_dim))

        # 3. 交叉注意力机制：将新信息融入记忆槽
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 4. 记忆更新门控（类似GRU）
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim), nn.Sigmoid()
        )

        # 5. 记忆解码器：将记忆映射回LLM空间
        self.memory_decoder = nn.Sequential(
            nn.Linear(memory_dim, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        # 6. 重要性评分网络（决定哪些记忆需要保留）
        self.importance_scorer = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Linear(memory_dim // 2, 1),
            nn.Sigmoid(),
        )

    def encode_experience(
        self, llm_hidden_states: torch.Tensor  # [batch, seq_len, llm_hidden_size]
    ) -> torch.Tensor:
        """将LLM的隐藏状态编码为记忆表示"""
        # 压缩到记忆空间
        memory_repr = self.memory_encoder(
            llm_hidden_states
        )  # [batch, seq_len, memory_dim]
        return memory_repr

    def update_memory(
        self,
        current_memory: torch.Tensor,  # [batch, num_slots, memory_dim]
        new_experience: torch.Tensor,  # [batch, seq_len, memory_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用交叉注意力更新记忆
        返回：更新后的记忆 + 重要性分数
        """
        batch_size = new_experience.shape[0]

        # 扩展记忆槽到batch
        if current_memory is None:
            current_memory = self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)

        # 交叉注意力：记忆槽作为query，新经验作为key/value
        attended_memory, attn_weights = self.cross_attention(
            query=current_memory,
            key=new_experience,
            value=new_experience,
            key_padding_mask=mask,
        )

        # 门控更新（类似GRU）
        gate_input = torch.cat([current_memory, attended_memory], dim=-1)
        update_gate = self.update_gate(gate_input)

        updated_memory = (
            update_gate * attended_memory + (1 - update_gate) * current_memory
        )

        # 计算记忆重要性分数
        importance_scores = self.importance_scorer(
            updated_memory
        )  # [batch, num_slots, 1]

        return updated_memory, importance_scores.squeeze(-1)

    def retrieve_memory(
        self,
        memory: torch.Tensor,  # [batch, num_slots, memory_dim]
        query_hidden_states: torch.Tensor,  # [batch, seq_len, llm_hidden_size]
    ) -> torch.Tensor:
        """
        检索相关记忆并映射回LLM空间
        """
        # 将query编码到记忆空间
        query_memory = self.memory_encoder(query_hidden_states)

        # 计算相似度（注意力）
        attn_scores = torch.bmm(query_memory, memory.transpose(1, 2)) / (
            self.memory_dim**0.5
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 检索记忆
        retrieved_memory = torch.bmm(
            attn_weights, memory
        )  # [batch, seq_len, memory_dim]

        # 解码回LLM空间
        memory_for_llm = self.memory_decoder(retrieved_memory)

        return memory_for_llm


class MemoryAugmentedLLMAgent(nn.Module):
    """
    记忆增强的LLM Agent
    """

    def __init__(
        self,
        pretrained_llm: nn.Module,
        llm_hidden_size: int = 4096,
        memory_config: dict = None,
    ):
        super().__init__()

        # 冻结预训练LLM
        self.llm = pretrained_llm
        for param in self.llm.parameters():
            param.requires_grad = False

        # 可训练的记忆模块
        memory_config = memory_config or {}
        self.memory_module = DynamicLatentMemoryCompressor(
            llm_hidden_size=llm_hidden_size, **memory_config
        )

        # 记忆状态（在推理时保持）
        self.register_buffer("current_memory", None)

        # 记忆注入层（将检索到的记忆融入LLM）
        self.memory_injection = nn.Linear(llm_hidden_size * 2, llm_hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        use_memory: bool = True,
    ):
        """
        前向传播
        """
        batch_size = input_ids.shape[0]

        # 1. 通过冻结的LLM获取隐藏状态
        with torch.no_grad():
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = llm_outputs.hidden_states[-1]  # 最后一层

        # 2. 检索记忆（如果存在）
        if use_memory and self.current_memory is not None:
            retrieved_memory = self.memory_module.retrieve_memory(
                self.current_memory, hidden_states
            )

            # 融合LLM隐藏状态和记忆
            combined = torch.cat([hidden_states, retrieved_memory], dim=-1)
            enhanced_hidden = self.memory_injection(combined)
        else:
            enhanced_hidden = hidden_states

        # 3. 更新记忆
        if update_memory:
            # 编码当前经验
            new_experience = self.memory_module.encode_experience(hidden_states)

            # 更新记忆槽
            updated_memory, importance_scores = self.memory_module.update_memory(
                self.current_memory,
                new_experience,
                mask=~attention_mask.bool() if attention_mask is not None else None,
            )

            # 根据重要性进行记忆压缩（可选）
            if self.training:
                # 训练时保留所有记忆用于梯度反传
                self.current_memory = updated_memory
            else:
                # 推理时可以根据重要性裁剪
                self.current_memory = updated_memory

        return {
            "enhanced_hidden_states": enhanced_hidden,
            "llm_logits": llm_outputs.logits,
            "memory_state": self.current_memory,
            "importance_scores": importance_scores if update_memory else None,
        }

    def reset_memory(self):
        """重置记忆状态"""
        self.current_memory = None


class MemoryTrainer:
    """
    记忆模块的训练器
    """

    def __init__(self, model: MemoryAugmentedLLMAgent, learning_rate: float = 1e-4):
        self.model = model

        # 只优化记忆模块参数
        memory_params = [
            p
            for n, p in model.named_parameters()
            if "memory_module" in n or "memory_injection" in n
        ]

        self.optimizer = torch.optim.AdamW(memory_params, lr=learning_rate)

    def compute_memory_loss(
        self,
        outputs: dict,
        targets: torch.Tensor,
        use_importance_regularization: bool = True,
    ):
        """
        计算记忆模块的损失
        """
        # 1. 主要任务损失（如语言建模）
        logits = outputs["llm_logits"]
        task_loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )

        # 2. 记忆重要性正则化（鼓励稀疏性）
        if use_importance_regularization and outputs["importance_scores"] is not None:
            importance_scores = outputs["importance_scores"]
            # L1正则化鼓励稀疏记忆
            sparsity_loss = importance_scores.mean()

            # 熵正则化鼓励记忆分化
            importance_dist = importance_scores / importance_scores.sum(
                dim=-1, keepdim=True
            )
            entropy = (
                -(importance_dist * torch.log(importance_dist + 1e-10))
                .sum(dim=-1)
                .mean()
            )
            entropy_loss = -entropy  # 最大化熵

            total_loss = task_loss + 0.01 * sparsity_loss + 0.001 * entropy_loss
        else:
            total_loss = task_loss

        return total_loss

    def train_step(self, batch):
        """单步训练"""
        self.optimizer.zero_grad()

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            update_memory=True,
            use_memory=True,
        )

        loss = self.compute_memory_loss(outputs, batch["labels"])

        loss.backward()
        self.optimizer.step()

        return loss.item()
