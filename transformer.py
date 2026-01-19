import torch
import torch.nn as nn
import math
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import torch.utils.data as data
from collections import Counter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 학습되는 파라미터가 아니므로 register_buffer 사용 (state_dict에 포함되지만 역전파 안됨)
        pe = torch.zeros(max_len, d_model)

        # 위치 인덱스 (0, 1, ... max_len-1)
        # shape: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 분모 계산 (10000^(2i/d_model))
        # log 공간에서 계산하여 수치적 안정성 확보: exp(log(10000) * -2i / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 짝수 인덱스(2i)에는 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 인덱스(2i+1)에는 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 배치 처리를 위해 차원 추가: [1, max_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: [Batch, Seq_Len, d_model]
        """
        # 입력 길이만큼 잘라서 더해줌 (Broadcasting)
        # x.size(1)은 현재 문장의 길이
        x = x + self.pe[:, :x.size(1), :]
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # 전체 차원 (예: 512)
        self.n_head = n_head   # 헤드 개수 (예: 8)
        self.d_head = d_model // n_head # 각 헤드의 차원 (예: 64)

        # d_model이 n_head로 나누어 떨어지는지 확인
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        # 1. W_Q, W_K, W_V: 입력을 Q, K, V로 변환하는 선형 층
        # (논문에서는 각 헤드별로 W를 따로 두지만, 구현상 통으로 만들고 나중에 쪼개는 것이 효율적)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 2. W_O: 여러 헤드의 결과를 합친 후 통과시키는 출력 선형 층
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        입력을 헤드 개수만큼 쪼개는 함수
        Input:  [Batch, Seq_Len, d_model]
        Output: [Batch, n_head, Seq_Len, d_head] -> Transpose로 헤드 차원을 앞으로
        """
        x = x.view(batch_size, -1, self.n_head, self.d_head)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        수식: Softmax(QK^T / sqrt(d_k)) * V
        """
        # 1. Dot Product (QK^T)
        # query: [Batch, n_head, Seq_Len, d_head]
        # key.transpose: [Batch, n_head, d_head, Seq_Len]
        # score: [Batch, n_head, Seq_Len, Seq_Len] (단어 간의 관계 점수)
        score = torch.matmul(query, key.transpose(-2, -1))

        # 2. Scaling (나누기 sqrt(d_k))
        scale = math.sqrt(self.d_head)
        score = score / scale

        # 3. Masking (옵션)
        # 마스크가 있다면 0인 위치(패딩 등)에 매우 작은 값(-1e9)을 넣어 Softmax 결과가 0이 되게 함
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 4. Softmax (확률 분포 생성)
        # dim=-1: 마지막 차원(Key의 시퀀스)에 대해 확률 계산
        attention_weights = torch.softmax(score, dim=-1)

        # 5. Weighted Sum (* V)
        # [Batch, n_head, Seq_Len, Seq_Len] @ [Batch, n_head, Seq_Len, d_head]
        # -> [Batch, n_head, Seq_Len, d_head]
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Linear Projection (Q, K, V 생성)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 2. Split Heads (헤드 나누기)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 3. Scaled Dot-Product Attention (핵심 연산)
        # attn_output: [Batch, n_head, Seq_Len, d_head]
        attn_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 4. Concat Heads (헤드 합치기)
        # Transpose 반대로: [Batch, Seq_Len, n_head, d_head]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # View로 합침: [Batch, Seq_Len, d_model]
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 5. Final Linear Projection (출력 층)
        output = self.w_o(attn_output)

        return output, attention_weights
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 1. 첫 번째 선형 변환 (확장: d_model -> d_ff)
        self.w_1 = nn.Linear(d_model, d_ff)
        # 2. 두 번째 선형 변환 (축소: d_ff -> d_model)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Linear -> ReLU -> Linear
        return self.w_2(self.dropout(self.relu(self.w_1(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 앞서 구현한 MultiHeadAttention 재사용
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Sub-layer 1: Self-Attention
        # 잔차 연결: x + Dropout(Attention(x))
        # Norm: LayerNorm(잔차연결 결과)
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Sub-layer 2: Feed Forward
        # 잔차 연결: x + Dropout(FFN(x))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_head)      # Masked Self-Attention
        self.cross_attn = MultiHeadAttention(d_model, n_head)     # Encoder-Decoder Attention
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x: 디코더 입력 (타겟 문장)
        # enc_output: 인코더의 최종 출력 (Key, Value로 사용)

        # 1. Masked Self-Attention (자신의 미래 단어 참조 금지)
        # Query=x, Key=x, Value=x
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross-Attention (인코더 정보 가져오기)
        # Query=디코더(x), Key=인코더(enc_output), Value=인코더(enc_output)
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 3. Feed Forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_head=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()

        # 임베딩 및 위치 인코딩
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model) # (이전 답변 코드 참조)
        self.dropout = nn.Dropout(dropout)

        # 인코더 층 쌓기 (ModuleList 활용)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])

        # 디코더 층 쌓기
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])

        # 최종 출력 층
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def generate_mask(self, src, tgt):
        # 1. 패딩 마스크 (0인 부분 가리기)
        # src: [Batch, Src_Len] -> [Batch, 1, 1, Src_Len] (Broadcasting 준비)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

        # 2. Look-ahead 마스크 (디코더용, 미래 가리기)
        tgt_len = tgt.size(1)
        # torch.tril: 하삼각 행렬 (대각선 아래만 1, 위는 0)
        nopeak_mask = torch.tril(torch.ones(tgt_len, tgt_len)).type(torch.bool).to(tgt.device)

        # 패딩 마스크와 Look-ahead 마스크 결합 (둘 다 True여야 보임)
        tgt_mask = tgt_pad_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # 마스크 생성
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # --- 인코더 과정 ---
        src = self.dropout(self.pos_encoding(self.src_embedding(src)))

        # 인코더 층 반복 통과
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # --- 디코더 과정 ---
        tgt = self.dropout(self.pos_encoding(self.tgt_embedding(tgt)))

        # 디코더 층 반복 통과 (인코더 출력인 enc_output을 계속 참조)
        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        # 최종 예측
        output = self.fc_out(dec_output)
        return output

# ===================================================================
# 학습 설정
# ===================================================================   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")  # 'cuda'가 나와야 GPU 사용 중

BATCH_SIZE = 128
MAX_LEN = 40  # 너무 길면 자름
SPECIAL_TOKENS = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}

# --- 단어장(Vocab) 생성 함수 ---
def build_vocab(sentences, min_freq=1):
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(str(sentence).split())
    
    vocab = SPECIAL_TOKENS.copy()
    idx = len(vocab)
    
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

dataset = pd.read_csv("./data/conversations.csv")
df = dataset.dropna(subset=['eng_sent', 'kor_sent'])

src_vocab_map = build_vocab(df['eng_sent'])
tgt_vocab_map = build_vocab(df['kor_sent'])
# print(src_vocab_map)

# ID -> 단어 역변환 사전 (나중에 결과 확인용)
tgt_inv_vocab = {v: k for k, v in tgt_vocab_map.items()}

print(f"영어 단어장 크기: {len(src_vocab_map)}")
print(f"한국어 단어장 크기: {len(tgt_vocab_map)}")

def encode(sentences, vocab, max_len):
    out = []
    for s in sentences:
        ids = [vocab.get(w, 3) for w in str(s).split()][:max_len-2]
        ids = [1] + ids + [2] + [0] * (max_len - len(ids) - 2)
        out.append(ids)
    return torch.LongTensor(out)

enc_inputs = encode(df['eng_sent'], src_vocab_map, MAX_LEN)
dec_outputs = encode(df['kor_sent'], tgt_vocab_map, MAX_LEN)

loader = data.DataLoader(data.TensorDataset(enc_inputs, dec_outputs), batch_size=BATCH_SIZE, shuffle=True)

model = Transformer(len(src_vocab_map), len(tgt_vocab_map), d_model=256, n_head=8, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 1. 모델 및 옵티마이저 설정
model = Transformer(len(src_vocab_map), len(tgt_vocab_map), d_model=256, n_head=8, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 2. 학습 루프
model.train()
for epoch in range(1, 51):
    total_loss = 0
    for src, tgt in loader:
        # 데이터 GPU 이동
        src, tgt = src.to(device), tgt.to(device)
        
        # 입력용과 정답용 분리
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # 기울기 초기화
        optimizer.zero_grad()
        
        # 순전파 (autocast 제외)
        output = model(src, tgt_in)
        
        # 손실 계산 및 역전파
        loss = criterion(output.reshape(-1, len(tgt_vocab_map)), tgt_out.reshape(-1))
        loss.backward()
        
        # 가중치 업데이트 (scaler 없이 바로 step)
        optimizer.step()
        
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

# ===================================================================
# 문장 넣어서 번역 확인하기
# ===================================================================
def translate(sentence, model, src_vocab, tgt_inv_vocab, max_len=40):
    model.eval() # 평가 모드 전환
    
    # 1. 입력 문장 전처리 (학습 시와 동일하게 <sos>, <eos>, padding 추가)
    # ids = [src_vocab.get(w, 3) for w in str(sentence).split()][:max_len-2]
    # ids = [1] + ids + [2] + [0] * (max_len - len(ids) - 2)
    # test_src = torch.LongTensor([ids]).to(device)

    # 1. 입력 문장 토큰화 및 ID 변환 (학습 시 encode 함수 로직과 동일)
    tokens = str(sentence).split()
    ids = [src_vocab.get(w, 3) for w in tokens][:max_len-2]
    ids = [1] + ids + [2] + [0] * (max_len - len(ids) - 2) # <sos> + ids + <eos> + pad
    test_src = torch.LongTensor([ids]).to(device)

    # 2. 디코더 입력 초기화 (<sos> 토큰으로 시작)
    test_tgt = torch.tensor([[1]]).to(device) # <sos>

    # 3. 한 단어씩 예측 (Greedy Decoding)
    for _ in range(max_len):
        with torch.no_grad():
            pred = model(test_src, test_tgt)
        
        # 마지막 타임스텝의 결과 중 확률이 가장 높은 단어 선택
        next_word = pred.argmax(dim=-1)[:, -1:]
        
        # 예측된 단어를 디코더 입력에 추가
        test_tgt = torch.cat([test_tgt, next_word], dim=-1)
        
        # <eos> (ID 2) 토큰이 나오면 멈춤
        if next_word.item() == 2:
            break

    # 4. ID를 단어로 변환 (결과 출력)
    # <sos>와 <eos>를 제외하고 출력
    decoded = [tgt_inv_vocab.get(i.item(), "?") for i in test_tgt[0, 1:] if i.item() not in [1, 2]]
    return " ".join(decoded)

# ==========================================
# 실제 사용 예시
# ==========================================
input_sentence = "Yes, how is it today?" # <--- 여기에 원하는 문장을 넣으세요.
result = translate(input_sentence, model, src_vocab_map, tgt_inv_vocab, MAX_LEN)

print(f"입력(Eng): {input_sentence}")
print(f"출력(Kor): {result}")