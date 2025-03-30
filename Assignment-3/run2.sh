CUDA_VISIBLE_DEVICES=1 python task2.py --hf-token "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" --decoding-strategy "single-head" --debug True
CUDA_VISIBLE_DEVICES=0 python task2.py --hf-token "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" --decoding-strategy "multi-head" --debug True --beam-width 2 --use-no-medusa-heads 2
CUDA_VISIBLE_DEVICES=1 python task2.py --hf-token "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" --decoding-strategy "multi-head" --debug True --beam-width 5 --use-no-medusa-heads 2
CUDA_VISIBLE_DEVICES=2 python task2.py --hf-token "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" --decoding-strategy "multi-head" --debug True --beam-width 10 --use-no-medusa-heads 2
CUDA_VISIBLE_DEVICES=0 python task2.py --hf-token "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" --decoding-strategy "multi-head" --debug True --beam-width 2 --use-no-medusa-heads 5
CUDA_VISIBLE_DEVICES=1 python task2.py --hf-token "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" --decoding-strategy "multi-head" --debug True --beam-width 5 --use-no-medusa-heads 5
CUDA_VISIBLE_DEVICES=2 python task2.py --hf-token "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" --decoding-strategy "multi-head" --debug True --beam-width 10 --use-no-medusa-heads 5