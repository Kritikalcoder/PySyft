from syft.generic.tensor import AbstractTensor
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.workers.abstract import AbstractWorker
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor

from torch.distributions import Laplace

import syft as sy
import torch as th


def min_tensor(*tensors):
    tensors = list(tensors)
    for i in range(len(tensors)):
        tensors[i] = tensors[i].unsqueeze(0)
    return th.cat(tensors).min(0).values


def max_tensor(*tensors):
    tensors = list(tensors)
    for i in range(len(tensors)):
        tensors[i] = tensors[i].unsqueeze(0)
    return th.cat(tensors).max(0).values


class SensitivityTensor(AbstractTensor):
    def __init__(
        self,
        accountant=None,
        l=None,
        h=None,
        el=None,
        eh=None,
        owner=None,
        id=None,
        tags=None,
        description=None,
    ):
        """Initializes a LoggingTensor, whose behaviour is to log all operations
        applied on it.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LoggingTensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        self.accountant = accountant

        if h is not None:
            self.h = h.float()
        else:
            self.h = h

        if l is not None:
            self.l = l.float()
        else:
            self.l = l

        if eh is not None:
            self.eh = eh.float()
        else:
            self.eh = eh

        if el is not None:
            self.el = el.float()
        else:
            self.el = el

        # TODO: account for exact value duplication.
        # x.expand(big_shape).publish() should cost the same as x.publish()

    @property
    def l_ex(self):
        """
        Taking l and matching it to the shape of e_l
        """
        return self.l.unsqueeze(-1).expand(self.el.shape)

    @property
    def h_ex(self):
        """
        Taking h and matching it to the shape of e_h
        """
        return self.h.unsqueeze(-1).expand(self.eh.shape)

    def __str__(self):
        out = "SensitivityTensor>" + str(self.child) + "\n"
        out += "\ts:" + str(self.sensitivity) + "\n"
        out += "\tl:" + str(self.l) + "\n"
        out += "\th:" + str(self.h) + "\n"
        try:
            out += "\tel:" + str(self.el).replace("\n", "\n           ") + "\n"
            out += "\teh:" + str(self.eh).replace("\n", "\n           ") + "\n"
        except:
            ""
        return out

    def __repr__(self):
        return self.__str__()

    def __add__(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you might need sometimes to specify
        some particular behaviour: so here's what to start from :)
        """
        # Replace all syft tensors with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "__add__", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "__add__")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("__add__", response, wrap_type=type(self))

        other = args[0]

        if isinstance(other, SensitivityTensor):

            # private - private addition
            l = self.l + other.l
            h = self.h + other.h
            el = self.el + other.el
            eh = self.eh + other.eh

        else:

            # private - public addition
            l = self.l + other
            h = self.h + other
            el = self.el
            eh = self.eh

        response.l = l
        response.h = h
        response.el = el
        response.eh = eh
        response.accountant = self.accountant

        return response

    def __sub__(self, *args, **kwargs):
        """
        Here is the version of the subtract method without the decorator: as you can see
        it is much more complicated. However you might need sometime to specify
        some particular behaviour: so here's what to start from :)
        """
        other = args[0]

        return self + (other * -1)

    def sum(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you might need sometimes to specify
        some particular behaviour: so here what to start from :)
        """
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "sum", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "sum")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("sum", response, wrap_type=type(self))

        l = self.l.sum(*args, **kwargs)
        h = self.h.sum(*args, **kwargs)
        el = self.el.sum(*args, **kwargs)
        eh = self.eh.sum(*args, **kwargs)

        response.l = l
        response.h = h
        response.el = el
        response.eh = eh
        response.accountant = self.accountant

        return response

    def __mul__(self, *args, **kwargs):
        """
        Here is the version of the product method without the decorator: as you can see
        it is much more complicated. However you might need some time to specify
        some particular behaviour: so here's what to start from :)
        """
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "__mul__", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "__mul__")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("__mul__", response, wrap_type=type(self))

        other = args[0]

        if isinstance(other, SensitivityTensor):
            # private private multiplication
            l = min_tensor(self.h * other.l, self.l * other.h, self.l * other.l)
            h = max_tensor(self.h * other.h, self.l * other.l, self.h * other.l, self.l * other.h)

            el = min_tensor(
                self.el * other.h_ex,  # done
                self.l_ex * other.eh,  # done
                self.eh * other.l_ex,  # done
                self.h_ex * other.el,  # done
                self.el * other.l_ex,
                self.l_ex * other.el,
            )

            eh = max_tensor(
                self.h_ex * other.eh,
                self.eh * other.h_ex,
                self.l_ex * other.el,
                self.el * other.h_ex,
            )

        else:

            l = self.l * other
            h = self.h * other
            el = self.el * other
            eh = self.eh * other

        response.l = l
        response.h = h
        response.el = el
        response.eh = eh
        response.accountant = self.accountant

        return response

    def __pow__(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "__pow__", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "__pow__")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("__pow__", response, wrap_type=type(self))

        exponent = args[0]

        # why is exponent even?
        assert exponent % 2 == 0

        l = self.l.pow(exponent)
        h = self.h.pow(exponent)
        el = self.el.pow(exponent)
        eh = self.eh.pow(exponent)

        response.l = l
        response.h = h
        response.el = el
        response.eh = eh
        response.accountant = self.accountant

        return response

    def sqrt(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "sqrt", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "sqrt")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("sqrt", response, wrap_type=type(self))

        l = self.l.sqrt(*args)
        h = self.h.sqrt(*args)
        el = self.el.sqrt(*args)
        eh = self.eh.sqrt(*args)

        response.l = l
        response.h = h
        response.el = el
        response.eh = eh
        response.accountant = self.accountant

        return response

    def mean(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you might need sometimes to specify
        some particular behaviour: so here what to start from :)
        """

        dim = args[0]
        dimlen = self.child.shape[dim]
        normalizer = 1 / dimlen

        return self.sum(*args, **kwargs) * normalizer

    def std(self, dim=0):
        return ((self - (self.mean(dim).unsqueeze(dim).expand(self.shape))) ** 2).mean(dim).sqrt()

    def __getitem__(self, *args, **kwargs):
        return (
            SensitivityTensor(
                self.accountant,
                self.l.__getitem__(*args, **kwargs),
                self.h.__getitem__(*args, **kwargs),
                self.el.__getitem__(*args, **kwargs),
                self.eh.__getitem__(*args, **kwargs),
            )
            .on(self.child.__getitem__(*args, **kwargs))
            .child
        )

    def unsqueeze(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "unsqueeze", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "unsqueeze")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("unsqueeze", response, wrap_type=type(self))

        if args[0] < 0:
            args[0] -= 1

        l = self.l.unsqueeze(*args, **kwargs)
        h = self.h.unsqueeze(*args, **kwargs)
        el = self.el.unsqueeze(*args, **kwargs)
        eh = self.eh.unsqueeze(*args, **kwargs)

        response.l = l
        response.h = h
        response.el = el
        response.eh = eh
        response.accountant = self.accountant

        return response

    def expand(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "expand", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "expand")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("expand", response, wrap_type=type(self))

        l = self.l.expand(*args, **kwargs)
        h = self.h.expand(*args, **kwargs)

        expanded_shape_w_entity_dim = args[0] + (self.el.shape[-1],)

        el = self.el.expand(*expanded_shape_w_entity_dim, **kwargs)
        eh = self.eh.expand(*expanded_shape_w_entity_dim, **kwargs)

        response.l = l
        response.h = h
        response.el = el
        response.eh = eh
        response.accountant = self.accountant

        return response

    def z_score(self, std=None, mean=None, epsilon=1):

        if std is None:
            std = self.std(0).publish(epsilon / 2).unsqueeze(0).expand(self.shape)
            epsilon = epsilon / 2

        if mean is None:
            mean = self.mean(0).unsqueeze(0).expand(self.shape)

        std_den = (1 / std).public_private(accountant=self.accountant).child

        num = self - mean.public_private(accountant=self.accountant).child

        return num * std_den

    @property
    def sensitivity(self):
        if self.eh is not None and self.el is not None:
            return (self.eh - self.el).max(-1).values
        else:
            return self.child.unsqueeze(-1).expand(self.shape + (self.accountant.n_entities,))

    @property
    def entities(self):
        return ((self.eh - self.el) > 0).float()

    def publish(self, epsilon=1):
        l = Laplace(loc=0, scale=self.sensitivity / epsilon)
        result = self.child + l.sample()
        return result

    # Module & Function overloading

    # We overload two torch functions:
    # - torch.add
    # - torch.nn.functional.relu

    @staticmethod
    @overloaded.module
    def torch(module):
        """
        We use the @overloaded.module to specify we're writing here
        a function which should overload the function with the same
        name in the <torch> module
        :param module: object which stores the overloading functions

        Note that we used the @staticmethod decorator as we're in a
        class
        """

        def add(x, y):
            """
            You can write the function to overload in the most natural
            way, so this will be called whenever you call torch.add on
            Logging Tensors, and the x and y you get are also Logging
            Tensors, so compared to the @overloaded.method, you see
            that the @overloaded.module does not hook the arguments.
            """
            print("Log function torch.add")
            return x + y

        # Just register it using the module variable
        module.add = add

        @overloaded.function
        def mul(x, y):
            """
            You can also add the @overloaded.function decorator to also
            hook arguments, ie all the LoggingTensor are replaced with
            their child attribute
            """
            print("Log function torch.mul")
            return x * y

        # Just register it using the module variable
        module.mul = mul

        # You can also overload functions in submodules!
        @overloaded.module
        def nn(module):
            """
            The syntax is the same, so @overloaded.module handles recursion
            Note that we don't need to add the @staticmethod decorator
            """

            @overloaded.module
            def functional(module):
                def relu(x):
                    print("Log function torch.nn.functional.relu")
                    return x * (x.child > 0)

                module.relu = relu

            module.functional = functional

        # Modules should be registered just like functions
        module.nn = nn

    @classmethod
    def on_function_call(cls, command):
        """
        Override this to perform a specific action for each call of a torch
        function with arguments containing syft tensors of the class doing
        the overloading
        """
        cmd, _, args, kwargs = command
        print("Default log", cmd)

    @staticmethod
    def simplify(tensor: "SensitivityTensor") -> tuple:
        """
        This function takes the attributes of a LogTensor and saves them in a tuple
        Args:
            tensor (LoggingTensor): a LogTensor
        Returns:
            tuple: a tuple holding the unique attributes of the log tensor
        Examples:
            data = _simplify(tensor)
        """

        chain = None
        if hasattr(tensor, "child"):
            chain = sy.serde._simplify(tensor.child)
        return (tensor.id, chain)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "LoggingTensor":
        """
        This function reconstructs a LogTensor given it's attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the LogTensor
        Returns:
            LoggingTensor: a LogTensor
        Examples:
            logtensor = detail(data)
        """
        obj_id, chain = tensor_tuple

        tensor = SensitivityTensor(owner=worker, id=obj_id)

        if chain is not None:
            chain = sy.serde._detail(worker, chain)
            tensor.child = chain

        return tensor
