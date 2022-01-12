There are different levels at which we want to log the loss values and perhaps other metrics.

```
   Model
     │
     ├──Sampler
     │     │
     │     └──Loss
     │
     │
     │
     ├──Inf_net
     │     │
           └──Loss
```
The main way of reporting the metrics is by calling the `get_metrics` method on the main model. When it is called, it will create a dictionary, fill it with the main metrics, and then it will call the corresponding method on `Sampler` and `Inf_net`. The `Sampler.get_metrics` and `Inf_net.get_metrics` will return their individual metrics in a possibly nested dictionary, which the `Model.get_metrics` will flatten and append to its output dict. This way of reporting metrics is mainly for the console log.

For more detailed reporting, we will use plots and callbacks.

# General logging framework

The `modules/logging.py` contains the classes required for logging. The main class is the `LoggingMixin` which should be used as one of the base classes for all modules/classes that require logging to be enable, for example, Model, Sampler and Loss. 

Each `LoggingMixin` will act as a node of a tree. The following figure shows two instances of `LoggingMixin`, *Parent-Node* and *Child-Node*.

```

  Parent-Node
        │
        ├────log-1
        │
        ├────log-2
        │ .
        │ .
        │
        └──Child-Node
               │
               ├───
               │
               ├───
               │
               .
               .
               .

                    Leaf-Node
                       │
                       ├────log-1
                       │
                       └────log-2

```
The two important concepts to note here are **Direct Logged Values** and **Children Nodes**.

### Direct Logged Values
As seen, a Node can have as many `log-x` or **direct logged values** as it wants. The direct logged values have to be registered in the constructor by adding an entry to the dictionary `self.logging_buffer: Dict[str, LoggedValue]`. The `LoggingMixin` will expose the `log()` method that can be used to log the registered **Direct logged values**.

### Children Nodes
Apart from Direct Logged Values, an instance of `LoggingMixin` or a Node can also have as many children instances of `LoggingMixin` or Nodes as required. The children nodes have to be registered by appending to the list `self.logging_children: List[LoggingMixin]`. The head Node will collect the logs from all Direct Logged Values and Children recursively on each call to `get_all()`. Hence, the `get_all()` method should only be called from the head node which in our case is the main model.

Ex:

See the code for `get_metrics()` in the base model, the constructor for inference_net sampler, the constructor of the base loss and the constructor for margin based loss.


