### Installing adapter package 
`pip install adapter-transformers` 

I strongly advise that you create another environment for this as we don't want other transformer/BERT models to be affected by this installation. FYI, what I did for this was simply copying current virtual env directory (e.g. `A` ) to new name you want (e.g. `B` ) and then modifying `B/bin/activate` file.

### How adapter loading works. 

Loading a new adapter is as follows:
*  `import transformer`
*  load pretrained BERT (or RoBERTa): 
```
                transformer.add_adapter(
                    adapter_task_name,
                    config=adapter_config_name
                )
```
where `adapter_task_name` is simply a string that you assign (e.g. 'scorenn', 'tasknn') and `adapter_config_name` is one of the following strings: "pfeiffer", "houlsby", "pfeiffer+inv", "houlsby+inv". For our research, these different types are not so important and we can simply use "pfeiffer". 

For our usage, we want two adapters for 'tasknn' and 'scorenn' and thus we will use 'scorenn' & 'tasknn' throughout this documentation.

### Making adapters participate.

* In making adapters participate in the forward/backward loop, we have to activate its participation. 
This is how it's usually done: `transformer.set_active_adapters(adapter_task_name)`

The usual usage is using one adapter at a time, but for our usage, we want both 'scorenn' adapter and 'tasknn' adapter to participate. Thus, we can use  [Parallel](https://docs.adapterhub.ml/adapter_composition.html?highlight=parallel#parallel) module in adapterhub as following:

`model.active_adapters = ac.Parallel(adapter1, adapter2)`

The [adapter-hub](https://github.com/Adapter-Hub/adapter-transformers/commit/e396d9c44b5beee86c28262921e2af38d0295674) recently added unit tests for Parallel option upon my request (nice!). 

### Choosing which adapter to backpropagate

`model.train_adapter(adapter_task_name)` only makes adapter parameters related to the `adapter_task_name` backpropagatable.

As we want to backpropagate to each adapter (scorenn, tasknn) in alternating fashion, we can simply do the following in the place where we used to activate `scorenn`, `tasknn` parameters exclusively.
```
# only finetune scorenn, freezing tasknn
model.train_adapter(scorenn)
...
# only finetune tasknn, freezing scorenn
model.train_adapter(tasknn)
```

### To work with AllenNLP

To work with allenNLP, I've written a wrapper files 
[cached_transformer_adapters.py](https://github.com/dhruvdcoder/structured_prediction_baselines/blob/feat/adapter/cached_transformer_adapters.py),
[pretrained_transformer_adpater_embedder](https://github.com/dhruvdcoder/structured_prediction_baselines/blob/feat/adapter/pretrained_transformer_adpater_embedder.py) in the `feat/adapter` branch.

This wrapper covers the basic addition, loading of adapters but does not handle `parallel` functionality nor freezing/unfreezing certain adapters. The latter needs to be taken care of inside the training loop and where to perform former is a design choice. 

### Saving & loading adpaters.
Saving and loading adapters can be done as shown in [tutorial](https://docs.adapterhub.ml/quickstart.html). I found it to be odd that you don't need the `adapter_task_name` (which in this example is 'sst-2') in loading the adapters, but I guess this it is what it is.
```
# save model
model.save_pretrained('./path/to/model/directory/')
# save adapter
model.save_adapter('./path/to/adapter/directory/', 'sst-2')

# load model
model = BertModel.from_pretrained('./path/to/model/directory/')
model.load_adapter('./path/to/adapter/directory/')
```







