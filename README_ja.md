# GatedNorm & PreAffineRMSNorm 実装

本リポジトリは、論文 **"A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is Essential for Transformer Training" (arXiv:2601.22966v1)** の概念をPyTorchで実装・検証したものです。

以下の手法を実装しました：
1.  **PreAffineRMSNorm**: RMSNormの前に学習可能なスケーリングベクトル $\lambda$ を導入し、Residual Sink（残差シンク）を吸収させる手法。
2.  **GatedNorm**: 正規化の後にLow-Rank Gating機構を追加し、明示的なリスケーリングを行う手法。

## 実験結果

**MiniGPT** モデル（Decoder-only Transformer, 約13Mパラメータ）を **WikiText-2** データセットで学習させ、性能を比較しました。

### 検証損失 (Validation Loss) - 低いほど良い

| 設定 | Val Loss | 備考 |
| :--- | :--- | :--- |
| **PreAffine** | **0.0465** | 最高性能（ベースラインをわずかに上回る）。 |
| **Baseline** (RMSNorm) | 0.0472 | 強力なベースライン。 |
| **GatedNorm** | 0.0496 | 性能は同等だが、わずかに損失が高い。 |

### 重み分析 (Weight Analysis)
論文が提唱する「Outlier-Driven Rescaling（外れ値駆動のリスケーリング）」仮説を裏付ける結果が得られました。

-   **Baseline**: RMSNormの重みが極端に0に近づく次元が見られ、巨大なResidual Sinkを抑制していることが示唆されます。
    ![Baseline Weights](results/weights_baseline_rms.png)

-   **PreAffine**: 学習可能な $\lambda$ ベクトルに巨大なスパイク（外れ値）が見られ、これがSinkを吸収する役割を果たしています。
    ![PreAffine Lambda](results/weights_preaffine_lambda.png)

-   **GatedNorm**: Gating機構により、Residual Stream（残差ストリーム）を明示的に制御しています。効果的ではありますが、PreAffineと比較してわずかに損失が高く、この規模ではより単純なPreAffineのスケーリングで十分、あるいは最適化が容易である可能性が示唆されます。
    ![GatedNorm Weights](results/weights_gatednorm_rms.png)

## 使用方法

### 必要要件
-   Python 3.10+
-   PyTorch
-   `uv` (推奨)

### インストール
```bash
git clone https://github.com/daichi202/gated-norm-experiment.git
cd gated-norm-experiment
uv sync
```

### 実験の実行
`experiment.py` を使用して学習および評価を行います。

```bash
# 全設定を実行
uv run experiment.py --config_name all --epochs 1 --batch_size 32

# 特定の設定のみ実行
uv run experiment.py --config_name PreAffine

# 学習済み重みをHugging Face Hubにアップロード
uv run experiment.py --push_to_hub --hf_repo_id "your-username/repo-name"
```

### HF Hubからの評価
Hugging Face Hub上のモデルをダウンロードして分析するには `evaluate.py` を使用します。

```bash
uv run evaluate.py --repo_id "daichi202/gated-norm-test"
```
