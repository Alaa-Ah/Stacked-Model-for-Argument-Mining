from src.transformers_ML.config import *
from src.transformers_ML.data_loader import RowSentencesHandler

def mapping(y):
    conversion = ['n', 'c', 'p']
    ans = [conversion[int(p)] for p in y]
    return ans

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def flat_classification_report(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print('classification report:')
    print(classification_report(labels_flat, preds_flat))


#########################################################
# Function to train the bert for sequence classification
# model
#########################################################
def TrainBertSeqCl(model, train_dataloader, optimizer, scheduler): #, class_weights):
    # Measure how long the training epoch takes.
    t0 = datetime.datetime.now()

    # Reset the total loss for this epoch.
    total_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = datetime.datetime.now() - t0
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids, 
                    # token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        # loss with class weights
        # outputs = model(b_input_ids, 
        #            token_type_ids=None, 
        #            attention_mask=b_input_mask)
        # b_logits = outputs[0]
        # cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        # loss = cross_entropy(b_logits, b_labels)


        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            

    # Store the loss value for plotting the learning curve.
    # loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(datetime.datetime.now() - t0))

    return avg_train_loss

#########################################################
# Function to evaluate the bert for sequence classification
# model
#########################################################
def EvaluateBertSeqCl(model, validation_dataloader):
    print("")
    print("Running Validation...")

    t0 = datetime.datetime.now()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    total_labels, total_preds = np.array([]), np.array([])
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                            # token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        total_labels = np.append(total_labels, label_ids.flatten())
        total_preds = np.append(total_preds, np.argmax(logits, axis=1).flatten())

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    # print("  Validation took: {:}".format(datetime.datetime.now() - t0))
    # print("  Accuracy: {0:.4f}".format(eval_accuracy / nb_eval_steps))
    # print("  Accuracy-2: {0:.4f}".format(flat_accuracy(total_preds, total_labels)))
    print(classification_report(total_labels, total_preds))

#########################################################
# Function to use the model for prediction
#########################################################
def Predict(model, sentences, logits_enable=False):
    model.eval()

    text_handler = RowSentencesHandler()
    dataloader = text_handler.GetDataLoader(sentences)
    prediction_result = np.array([])
    logits_result = np.array([[0,0]])

    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        prediction_result = np.append(prediction_result, np.argmax(logits, axis=1).flatten())
        logits_result = np.append(logits_result, logits, axis=0)

    if logits_enable:
        return logits_result[1:].tolist()

    return prediction_result.tolist()