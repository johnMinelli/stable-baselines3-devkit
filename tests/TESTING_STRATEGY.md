# Testing Strategy & Coverage

## Test Pyramid

```
                    _____
                   /     \
                  /  E2E  \         Slow, Real Envs Required
                 /_________\
                /           \
               / Integration \       Medium, Some Mocks
              /_______________\
             /                 \
            /    Unit Tests     \    Fast, All Mocked
           /_____________________\
```

## Coverage Levels

### Level 1: Unit Tests (Fast, No Dependencies) ✅
**Dependencies:** None (all mocked)
**Coverage:** ~85% of components

### Level 2: Integration Tests (Partial Dependencies) ⚠️
**Dependencies:** Some real imports, mostly mocks
**Coverage:** ~60% of integrations

### Level 3: Script Integration Tests (Slower, Real Dependencies) 🔴
**Dependencies:** Real environments/datasets required
**Coverage:** Actual entry points

## What Guarantees What?

### ✅ Unit Tests Pass → Core components work in isolation
- Preprocessors normalize correctly
- Policies have correct shapes (mock data)
- Buffers work properly
  
### ⚠️ Integration Tests Pass → Components work together (with mocks)
- Agents can use policies (with dummy envs)
- Data flows correctly from Preprocessor → Policy
- Mock training loops work

### ✅ Script Integration Tests Pass → README examples work!
- `train.py` commands from README execute successfully
- `train_off.py` works with real datasets
- `predict.py` loads and runs models
- Full train→save→load→predict workflow works
