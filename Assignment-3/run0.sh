CUDA_VISIBLE_DEVICES=1 python task0.py --hf-token "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" --decoding-strategy "greedy" --debug True

CUDA_VISIBLE_DEVICES=1 python task0.py --hf-token "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" --decoding-strategy "random" --tau 0.5 --debug True
CUDA_VISIBLE_DEVICES=1 python task0.py --hf-token "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" --decoding-strategy "random" --tau 0.9 --debug True

CUDA_VISIBLE_DEVICES=1 python task0.py --hf-token "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" --decoding-strategy "topk" --k 5 --debug True
CUDA_VISIBLE_DEVICES=1 python task0.py --hf-token "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" --decoding-strategy "topk" --k 10 --debug True

CUDA_VISIBLE_DEVICES=1 python task0.py --hf-token "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" --decoding-strategy "nucleus" --p 0.5 --debug True
CUDA_VISIBLE_DEVICES=1 python task0.py --hf-token "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" --decoding-strategy "nucleus" --p 0.9 --debug True