"""Small deterministic checks for the mathematical appendices. CPU, no datasets."""
from itertools import product
import json
import math
from pathlib import Path
import statistics
import torch

torch.set_num_threads(1)
here = Path(__file__).resolve().parent
results = {}


def close(value, expected, tolerance=1e-9):
    assert abs(float(value) - float(expected)) < tolerance, (value, expected)


def matrix_check():
    dtype = torch.float64
    x = torch.tensor([[1., 2.], [-1., 1.]], dtype=dtype, requires_grad=True)
    w = torch.tensor([[.5, -.5], [1., .25]], dtype=dtype, requires_grad=True)
    b = torch.tensor([.1, -.2], dtype=dtype, requires_grad=True)
    v = torch.tensor([.2, -.4], dtype=dtype, requires_grad=True)
    c = torch.tensor(.05, dtype=dtype, requires_grad=True)
    y = torch.tensor([.3, -.1], dtype=dtype)

    def loss_at(parameters):
        w1, b1, v1, c1 = parameters
        return ((torch.relu(x @ w1.T + b1) @ v1 + c1 - y).square()).mean() / 2

    parameters = [w, b, v, c]
    loss = loss_at(parameters)
    close(loss.item(), .15385)
    loss.backward()
    expected = [torch.tensor([[0., 0.], [.154, .308]], dtype=dtype),
                torch.tensor([0., .154], dtype=dtype),
                torch.tensor([0., -.5005], dtype=dtype), torch.tensor(-.31, dtype=dtype)]
    max_error = 0.
    for parameter, analytic in zip(parameters, expected):
        assert torch.allclose(parameter.grad, analytic, atol=1e-12, rtol=0)
    assert torch.allclose(x.grad, torch.tensor([[.154, .0385], [0., 0.]], dtype=dtype), atol=1e-12, rtol=0)
    for parameter_index, parameter in enumerate(parameters):
        for offset in range(parameter.numel()):
            plus = [p.detach().clone() for p in parameters]
            minus = [p.detach().clone() for p in parameters]
            plus[parameter_index].reshape(-1)[offset] += 1e-6
            minus[parameter_index].reshape(-1)[offset] -= 1e-6
            numeric = (loss_at(plus).item() - loss_at(minus).item()) / 2e-6
            error = abs(numeric - parameter.grad.reshape(-1)[offset].item())
            max_error = max(max_error, error)
            assert error < 1e-8
    updated = [p.detach() - .1 * p.grad for p in parameters]
    new_loss = loss_at(updated).item()
    assert new_loss < loss.item()
    jacobian = torch.tensor([[4., 1.], [3., 2.]], dtype=dtype)
    gradient = torch.tensor([.5, -1.], dtype=dtype)
    direction = torch.tensor([1., -1.], dtype=dtype)
    assert torch.allclose(jacobian.T @ gradient, torch.tensor([-1., -1.5], dtype=dtype))
    close(gradient @ jacobian @ direction, .5)
    results['A'] = dict(initial_loss=loss.item(), updated_loss=new_loss,
                        max_finite_difference_error=max_error, directional_derivative=.5)


def probability_check():
    p = .75
    likelihood = p**3 * (1 - p)
    nll = -3 * math.log(p) - math.log(1 - p)
    close(likelihood, .10546875)
    q = [.7, .2, .1]
    prediction = [.6, .3, .1]
    entropy = -sum(a * math.log(a) for a in q)
    cross_entropy = -sum(a * math.log(b) for a, b in zip(q, prediction))
    kl = sum(a * math.log(a / b) for a, b in zip(q, prediction))
    close(cross_entropy, entropy + kl)
    def gaussian_nll(variance):
        return sum(error**2/(2*variance) + .5*math.log(2*math.pi*variance) for error in [1.,3.]) / 2
    h = 1e-5
    derivative = (gaussian_nll(5+h)-gaussian_nll(5-h))/(2*h)
    close(derivative, 0., 1e-8)
    weighted_sum = sum(a*b for a,b in zip([.2,.6,1.], [1.,2.,1.]))
    close(weighted_sum, 2.4)
    results['B'] = dict(likelihood=likelihood, nll=nll, mean_nll=nll/4,
                        entropy=entropy, cross_entropy=cross_entropy, kl=kl,
                        variance_optimum_derivative=derivative,
                        weighted_mean_by_count=weighted_sum/3, weighted_mean_by_weight=weighted_sum/4)


def metrics_check():
    matrix = [[40,8,2],[5,25,0],[6,4,10]]
    f1 = []
    for k in range(3):
        tp = matrix[k][k]
        fp = sum(row[k] for row in matrix)-tp
        fn = sum(matrix[k])-tp
        f1.append(2*tp/(2*tp+fp+fn))
    macro = statistics.mean(f1)
    weighted = sum(value*sum(row) for value,row in zip(f1,matrix))/100
    # Compare with the six decimal places published in the manuscript.
    close(macro, .721116, 5e-7)
    close(weighted, .744920, 5e-7)
    scores = [.95,.85,.75,.65,.55,.45,.35,.25,.15,.05]
    labels = [1,0,1,1,0,0,1,0,0,0]
    threshold_results = {}
    for threshold in [.5,.3]:
        predictions = [s >= threshold for s in scores]
        tp = sum(p and y for p,y in zip(predictions,labels))
        fp = sum(p and not y for p,y in zip(predictions,labels))
        fn = sum(not p and y for p,y in zip(predictions,labels))
        threshold_results[str(threshold)] = dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn))
    differences = [1,2,3,4,5]
    observed = statistics.mean(differences)
    extremes = sum(abs(statistics.mean(a*b for a,b in zip(signs,differences))) >= observed-1e-12
                   for signs in product([-1,1], repeat=5))
    close(extremes/32, .0625)
    def wilson(k,n,z=1.96):
        p = k/n
        denominator = 1+z*z/n
        center = (p+z*z/(2*n))/denominator
        half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
        return [center-half,center+half]
    results['D'] = dict(f1=f1,macro_f1=macro,weighted_f1=weighted,
                        thresholds=threshold_results,sd=statistics.stdev(differences),
                        standard_error=statistics.stdev(differences)/math.sqrt(5),
                        exact_sign_p=extremes/32,wilson80=wilson(80,100),wilson100=wilson(100,100))


def checkpoint_check():
    torch.manual_seed(2026)
    x = torch.tensor([[1.,2.],[-1.,1.],[0.,-1.]])
    y = torch.tensor([[.3],[-.1],[.4]])
    model = torch.nn.Linear(2,1)
    optimizer = torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
    def step(m,o):
        o.zero_grad()
        loss = (m(x)-y).square().mean()
        loss.backward()
        o.step()
        return loss.item()
    step(model,optimizer)
    step(model,optimizer)
    checkpoint_file = here/'checkpoint_verification.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(),step=2),checkpoint_file)
    expected_random = torch.rand(4)
    step(model,optimizer)
    uninterrupted = {k:v.detach().clone() for k,v in model.state_dict().items()}
    state = torch.load(checkpoint_file,map_location='cpu',weights_only=True)
    restored = torch.nn.Linear(2,1)
    restored_optimizer = torch.optim.SGD(restored.parameters(),lr=.05,momentum=.9)
    restored.load_state_dict(state['model'])
    restored_optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    assert torch.equal(torch.rand(4),expected_random)
    step(restored,restored_optimizer)
    full_error = max((uninterrupted[k]-v).abs().max().item() for k,v in restored.state_dict().items())
    assert full_error == 0
    weights_only_model = torch.nn.Linear(2,1)
    weights_only_model.load_state_dict(state['model'])
    fresh_optimizer = torch.optim.SGD(weights_only_model.parameters(),lr=.05,momentum=.9)
    step(weights_only_model,fresh_optimizer)
    missing_momentum_error = max((uninterrupted[k]-v).abs().max().item() for k,v in weights_only_model.state_dict().items())
    assert missing_momentum_error > 0
    checkpoint_file.unlink()
    results['E'] = dict(full_resume_error=full_error,weights_only_next_step_difference=missing_momentum_error,
                        rng_restored=True,torch=str(torch.__version__))


matrix_check()
probability_check()
metrics_check()
checkpoint_check()
output = dict(status='PASS',results=results)
(here/'appendix_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2))
