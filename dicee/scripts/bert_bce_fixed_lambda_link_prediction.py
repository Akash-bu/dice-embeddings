"""Train the joint BERT/KGE BCE model with a fixed CLI-supplied lambda.

This entry point reuses the joint training and evaluation pipeline from
``bert_bce_link_prediction.py`` while preventing lambda from receiving
gradients or being included in the optimizer.
"""

from dicee.scripts.bert_bce_link_prediction import main


if __name__ == "__main__":
    main(use_fixed_lambda=True)
