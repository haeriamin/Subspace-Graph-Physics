#!/bin/bash

c=1
# Depths [m]
for d in $(echo "scale=2; 0.02" | bc -l) \
         $(echo "scale=2; 0.04" | bc -l) \
         $(echo "scale=2; 0.05" | bc -l) \
         $(echo "scale=2; 0.08" | bc -l) \
         $(echo "scale=2; 0.10" | bc -l)
do
    # Horizontal speeds [m/s]
    for s in $(echo "scale=2; 0.01" | bc -l) \
             $(echo "scale=2; 0.04" | bc -l) \
             $(echo "scale=2; 0.08" | bc -l) \
             $(echo "scale=2; 0.10" | bc -l) \
             $(echo "scale=2; 0.15" | bc -l)
    do
        # Angles [deg]
        for a in 0.0 3.8 10.0 30.8 45.0
        do
            # Motion types
            for m in 1.0 2.0
            do
                # Skip 50 test datasets:
                # 2, 10, 14, 16, 22, 23, 37, 48, 52, 57,
                # 65, 73, 77, 79, 81, 83, 84, 90, 94, 97,
                # 99, 101, 102, 103, 114, 115, 116, 122, 126, 127,
                # 129, 132, 134, 137, 142, 165, 186, 198, 202, 205,
                # 210, 217, 223, 227, 229, 237, 238, 240, 242, 248
                if [[ ($c -ne 2) || ($c -ne 10) || ($c -ne 14) || ($c -ne 16) || ($c -ne 22) || \
                      ($c -ne 23) || ($c -ne 37) || ($c -ne 48) || ($c -ne 52) || ($c -ne 57) || \
                      ($c -ne 65) || ($c -ne 73) || ($c -ne 77) || ($c -ne 79) || ($c -ne 81) || \
                      ($c -ne 83) || ($c -ne 84) || ($c -ne 90) || ($c -ne 94) || ($c -ne 97) || \
                      ($c -ne 99) || ($c -ne 101) || ($c -ne 102) || ($c -ne 103) || ($c -ne 114) || \
                      ($c -ne 115) || ($c -ne 116) || ($c -ne 122) || ($c -ne 126) || ($c -ne 127) || \
                      ($c -ne 129) || ($c -ne 132) || ($c -ne 134) || ($c -ne 137) || ($c -ne 142) || \
                      ($c -ne 165) || ($c -ne 186) || ($c -ne 198) || ($c -ne 202) || ($c -ne 205) || \
                      ($c -ne 210) || ($c -ne 217) || ($c -ne 223) || ($c -ne 227) || ($c -ne 229) || \
                      ($c -ne 237) || ($c -ne 238) || ($c -ne 240) || ($c -ne 242) || ($c -ne 248) ]]
                then
                    path="./learning_to_simulate/datasets/Excavation/${c}_D0${d}m_S0${s}ms-1_A${a}deg_M${m}"
                    echo "${path}"
                    python -m learning_to_simulate.train \
                    --mode=train \
                    --eval_split=train \
                    --data_path=$path \
                    --model_path=./learning_to_simulate/models/Excavation
                fi
                c=`expr $c + 1`
            done
        done
    done
done

# Instructions:
# Mark it executable using:
# $ chmod +x run_excavation.sh
# Then run it:
# $ ./learning_to_simulate/run_excavation.sh