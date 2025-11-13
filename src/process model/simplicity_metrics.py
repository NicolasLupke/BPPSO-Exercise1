import math
from typing import Any, Iterable, TYPE_CHECKING
from pm4py import PetriNet

if TYPE_CHECKING:  # pragma: no cover - only for static analysis
    from pm4py.objects.petri_net.obj import PetriNet  # type: ignore[import]


def _ensure_petri_net(petri_net: Any) -> Any:
    if petri_net is None:
        raise ValueError("A Petri net instance is required to compute simplicity metrics.")
    for attr in ("places", "transitions", "arcs"):
        if not hasattr(petri_net, attr):
            raise TypeError(
                f"The provided object does not expose the expected '{attr}' attribute."
            )
    return petri_net


def _outgoing_transitions(place: Any) -> Iterable[Any]:
    for arc in getattr(place, "out_arcs", []):
        target = getattr(arc, "target", None)
        if target is not None and hasattr(target, "in_arcs"):
            yield target


def entropy_simplicity_petri_net(net: PetriNet):
    """
    Compute entropy-based simplicity metric for a PM4Py Petri net.

    Args:
        net (PetriNet): A PM4Py Petri net object.

    Returns:
        dict: {'entropy': H, 'simplicity': S}
    """
    if not isinstance(net, PetriNet):
        raise TypeError("Input must be a pm4py.objects.petri.petrinet.PetriNet object")

    places = list(net.places)
    transitions = list(net.transitions)
    total_entropy = 0.0
    place_count = len(places)

    for place in places:
        # Find outgoing transitions (place -> transition)
        outgoing_transitions = [arc.target for arc in place.out_arcs if isinstance(arc.target, PetriNet.Transition)]
        
        if not outgoing_transitions:
            continue

        # Assume uniform probability among outgoing transitions
        n = len(outgoing_transitions)
        probs = [1.0 / n] * n
        H_p = -sum(p * math.log2(p) for p in probs)
        total_entropy += H_p

    # Mean entropy per place
    H = total_entropy / place_count if place_count > 0 else 0
    H_max = math.log2(len(transitions)) if len(transitions) > 1 else 1
    S = 1 - (H / H_max if H_max > 0 else 0)

    return {'entropy': H, 'simplicity': S}

def size_simplicity_metric(petri_net: Any) -> int:
    """
    Simple size metric counting the structural elements of the net.
    Larger values indicate a less simple model.
    """
    net = _ensure_petri_net(petri_net)
    return len(net.places) + len(net.transitions) + len(net.arcs)

