# Principles for Concise Technical Writing in Research Papers

# 논문에서 간결한 테크니컬 라이팅을 위한 원칙

## 1. Put only one claim in each sentence.

## 한 문장에는 하나의 주장만 넣어라.

**English:**
A technical sentence should deliver one clear idea. If a sentence contains too many claims, split it into multiple sentences.

**Korean:**
기술적인 문장은 하나의 명확한 아이디어만 전달해야 한다. 한 문장 안에 주장이 너무 많이 들어가면 여러 문장으로 나누는 것이 좋다.

**Bad:**
We propose a novel SAE-based test-time adaptation method that improves robustness by identifying interpretable latent factors and adapting the model under distribution shift without requiring source data.

**Korean:**
우리는 해석 가능한 latent factor를 찾고, source data 없이 distribution shift 상황에서 모델을 적응시켜 robustness를 향상시키는 새로운 SAE 기반 TTA 방법을 제안한다.
→ 너무 많은 정보가 한 문장에 들어가 있다.

**Better:**
We propose an SAE-based test-time adaptation method.
The method identifies latent factors that are sensitive to distribution shift.
It then adapts the model using these factors without accessing source data.

**Korean:**
우리는 SAE 기반 test-time adaptation 방법을 제안한다.
이 방법은 distribution shift에 민감한 latent factor를 식별한다.
그 후 source data에 접근하지 않고, 해당 factor를 이용해 모델을 적응시킨다.

---

## 2. Replace adjectives with concrete conditions, observations, or numbers.

## 형용사보다 구체적인 조건, 관찰, 수치를 써라.

**English:**
Avoid vague adjectives such as significant, effective, meaningful, and robust unless they are supported by measurements.

**Korean:**
`significant`, `effective`, `meaningful`, `robust` 같은 모호한 형용사는 측정 결과로 뒷받침되지 않으면 피하는 것이 좋다.

**Weak:**
Our method significantly improves robustness.

**Korean:**
우리 방법은 robustness를 크게 향상시킨다.
→ 얼마나 향상됐는지 알 수 없다.

**Better:**
Our method improves OOD accuracy by 3.2% on ImageNet-R.

**Korean:**
우리 방법은 ImageNet-R에서 OOD accuracy를 3.2% 향상시킨다.
→ 구체적인 수치가 있어서 더 기술적이다.

---

## 3. Separate what you did from what you claim.

## 수행한 것과 주장하는 것을 분리해라.

**English:**
A method description should state what is done. A hypothesis or interpretation should be written separately.

**Korean:**
방법 설명에서는 실제로 무엇을 했는지를 써야 한다. 가설이나 해석은 별도의 문장으로 분리하는 것이 좋다.

**Weak:**
We use SAE latents to improve TTA because they capture meaningful concepts.

**Korean:**
우리는 SAE latent가 의미 있는 concept을 포착하기 때문에 이를 TTA 향상에 사용한다.
→ 방법과 가설이 섞여 있다.

**Better:**
We train an SAE on intermediate ViT features.
We use the resulting latent activations as adaptation signals at test time.
Our hypothesis is that these latents capture shift-sensitive factors.

**Korean:**
우리는 ViT의 intermediate feature 위에 SAE를 학습한다.
그 결과 얻어진 latent activation을 test time에서 adaptation signal로 사용한다.
우리의 가설은 이러한 latent가 shift-sensitive factor를 포착한다는 것이다.

---

## 4. Control the strength of your claims.

## 주장 강도를 조절해라.

**English:**
Do not overclaim unless the evidence is strong. Words such as prove, solve, causal, universal, and interpretable can invite strong criticism.

**Korean:**
근거가 충분하지 않다면 과하게 주장하지 않는 것이 좋다. `prove`, `solve`, `causal`, `universal`, `interpretable` 같은 단어는 리뷰어의 강한 비판을 받을 수 있다.

**Too strong:**
SAE latents provide interpretable causal factors for OOD robustness.

**Korean:**
SAE latent는 OOD robustness를 위한 해석 가능한 causal factor를 제공한다.
→ causal과 interpretable을 모두 강하게 주장하고 있어서 위험하다.

**Safer:**
SAE latents provide a structured representation for analyzing shift-sensitive factors.

**Korean:**
SAE latent는 shift-sensitive factor를 분석하기 위한 구조화된 representation을 제공한다.
→ 더 안전하고 방어 가능한 표현이다.

**Also safe:**
Our results suggest that some SAE latents are associated with OOD-sensitive visual factors.

**Korean:**
우리 결과는 일부 SAE latent가 OOD에 민감한 시각적 factor와 관련되어 있음을 시사한다.
→ “증명한다”가 아니라 “시사한다”라고 표현해서 주장 강도를 낮춘다.

---

## 5. Make the first sentence of each paragraph state its role.

## 각 문단의 첫 문장은 문단의 역할을 알려줘야 한다.

**English:**
The first sentence should tell the reader why the paragraph exists.

**Korean:**
문단의 첫 문장은 독자에게 이 문단이 왜 필요한지 알려줘야 한다.

**Example:**
First, we evaluate whether SAE latents capture perturbation-sensitive factors.
Second, we test whether these latents can guide test-time adaptation.
Finally, we analyze the failure cases of the proposed method.

**Korean:**
첫째, SAE latent가 perturbation-sensitive factor를 포착하는지 평가한다.
둘째, 이러한 latent가 test-time adaptation을 guide할 수 있는지 실험한다.
마지막으로, 제안한 방법의 실패 사례를 분석한다.

---

## 6. Avoid vague pronouns such as “this” and “it.”

## “this”, “it” 같은 모호한 대명사를 줄여라.

**English:**
In technical writing, repeating the key noun is often better than using an unclear pronoun.

**Korean:**
기술 논문에서는 모호한 대명사를 쓰는 것보다 핵심 명사를 반복하는 것이 더 좋을 때가 많다.

**Weak:**
This improves robustness under such shifts.

**Korean:**
이것은 그러한 shift에서 robustness를 향상시킨다.
→ “This”가 무엇을 가리키는지 불명확하다.

**Better:**
Using SAE latent activations as adaptation signals improves robustness under these shifts.

**Korean:**
SAE latent activation을 adaptation signal로 사용하는 것은 이러한 shift 상황에서 robustness를 향상시킨다.
→ 주어가 명확하다.

---

## 7. Describe experiments in a reproducible order.

## 실험 설명은 재현 가능한 순서로 써라.

**English:**
A good experimental description usually follows this order: dataset, model, feature layer, method setting, and metric.

**Korean:**
좋은 실험 설명은 보통 dataset, model, feature layer, method setting, metric 순서로 쓴다.

**Example:**
We evaluate ResNet-50 and ViT-B/16 on ImageNet and ImageNet-R.
For each model, we extract intermediate features from block 9.
We train an SAE with an expansion factor of 8 and an L1 coefficient of 1e-4.
We report clean accuracy, corrupted accuracy, and relative accuracy.

**Korean:**
우리는 ResNet-50과 ViT-B/16을 ImageNet 및 ImageNet-R에서 평가한다.
각 모델에 대해 block 9에서 intermediate feature를 추출한다.
우리는 expansion factor 8, L1 coefficient 1e-4로 SAE를 학습한다.
clean accuracy, corrupted accuracy, relative accuracy를 보고한다.

---

## 8. Explain novelty by stating the difference, not by exaggerating.

## novelty는 과장하지 말고 차이를 명확히 써라.

**English:**
Do not simply say that your method is better. Explain what previous methods do and what your method does differently.

**Korean:**
단순히 우리 방법이 더 좋다고 쓰지 말고, 기존 방법이 무엇을 하고 우리 방법이 무엇을 다르게 하는지 설명해야 한다.

**Weak:**
Unlike previous methods, our method is more effective and practical.

**Korean:**
기존 방법과 달리, 우리 방법은 더 효과적이고 실용적이다.
→ 구체적인 차이가 없다.

**Better:**
Unlike prior TTA methods that update batch normalization statistics or model parameters directly, our method uses SAE latent activations as the adaptation signal.

**Korean:**
batch normalization statistics나 model parameter를 직접 업데이트하는 기존 TTA 방법과 달리, 우리 방법은 SAE latent activation을 adaptation signal로 사용한다.
→ 기존 방법과의 차이가 명확하다.

---

## 9. Use simple sentence patterns.

## 단순한 문장 패턴을 사용해라.

**English:**
Technical writing does not need complex sentence structures. Simple and repeated patterns are often clearer.

**Korean:**
기술적 글쓰기에는 복잡한 문장 구조가 필요하지 않다. 단순하고 반복적인 패턴이 오히려 더 명확하다.

**Useful patterns:**
We propose X for Y.
X consists of A and B.
Given input x, we first compute z using f.
We then apply g to obtain h.
This design allows the model to use A without requiring B.
We evaluate X on D using M.
The results show that X improves Y under Z.
However, X is less effective when Z is severe.

**Korean:**
우리는 Y를 위한 X를 제안한다.
X는 A와 B로 구성된다.
입력 x가 주어졌을 때, 우리는 먼저 f를 사용해 z를 계산한다.
그 다음 g를 적용해 h를 얻는다.
이 설계는 B 없이 A를 사용할 수 있게 한다.
우리는 D에서 M을 사용해 X를 평가한다.
결과는 X가 Z 상황에서 Y를 향상시킴을 보여준다.
그러나 Z가 심할 때 X의 효과는 감소한다.

---

## 10. For your SAE-TTA research, describe observations rather than abstract interpretability.

## SAE-TTA 연구에서는 추상적인 interpretability보다 관찰 결과를 써라.

**English:**
Instead of saying that the SAE latents are interpretable, describe how specific latents respond to specific perturbations.

**Korean:**
SAE latent가 interpretable하다고 말하기보다, 특정 latent가 특정 perturbation에 어떻게 반응했는지를 쓰는 것이 좋다.

**Weak:**
The SAE latents are interpretable.

**Korean:**
SAE latent는 해석 가능하다.
→ 근거가 부족하고 공격받기 쉽다.

**Better:**
Several SAE latents show consistent activation changes under texture suppression and remain stable under color perturbation.

**Korean:**
일부 SAE latent는 texture suppression 상황에서 일관된 activation 변화를 보이며, color perturbation에서는 안정적으로 유지된다.
→ 구체적인 관찰을 기반으로 해서 더 논문답다.

---

## Final checklist

## 최종 체크리스트

**English:**
Does each sentence contain only one claim?
Is the subject clear?
Can vague adjectives be replaced with numbers or observations?
Are the method and interpretation separated?
Is the claim too strong for the evidence?
Can the statement be connected to an experiment, figure, or metric?

**Korean:**
각 문장에 하나의 주장만 들어 있는가?
주어가 명확한가?
모호한 형용사를 수치나 관찰로 바꿀 수 있는가?
방법과 해석이 분리되어 있는가?
근거에 비해 주장이 너무 강하지 않은가?
해당 문장을 실험, figure, metric과 연결할 수 있는가?

# 정리본
원칙1: **한 문장에는 하나의 주장만 넣어라.**
문장이 여러 주장, 방법, 결과, 의의를 동시에 담고 있으면 여러 문장으로 나눠라.

원칙2: **형용사보다 구체적인 조건, 관찰, 수치를 써라.**
`significant`, `effective`, `robust` 같은 표현은 가능하면 정확한 수치나 실험 결과로 바꿔라.

원칙3: **수행한 것과 주장하는 것을 분리해라.**
방법 설명에서는 실제로 한 일을 쓰고, 가설이나 해석은 별도 문장으로 분리해라.

원칙4: **주장 강도를 조절해라.**
근거가 충분하지 않다면 `prove`, `causal`, `universal`, `interpretable` 같은 강한 표현을 피하고 더 방어 가능한 표현을 사용해라.

원칙5: **각 문단의 첫 문장은 문단의 역할을 알려줘야 한다.**
문단의 첫 문장에서 이 문단이 평가, 방법 설명, 분석, 실패 사례 중 무엇을 다루는지 명확히 밝혀라.

원칙6: **“this”, “it” 같은 모호한 대명사를 줄여라.**
대명사 대신 핵심 명사를 반복해서 문장의 주어와 대상이 명확하게 보이도록 써라.

원칙7: **실험 설명은 재현 가능한 순서로 써라.**
dataset → model → feature layer → method setting → metric 순서로 설명하면 독자가 실험을 따라가기 쉽다.

원칙8: **novelty는 과장하지 말고 차이를 명확히 써라.**
기존 방법이 무엇을 하는지 먼저 쓰고, 우리 방법이 무엇을 다르게 하는지 구체적으로 설명해라.

원칙9: **단순한 문장 패턴을 사용해라.**
복잡한 문장보다 `We propose X`, `X consists of A and B`, `We evaluate X on D` 같은 단순한 구조를 반복하는 것이 더 명확하다.
