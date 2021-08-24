def wer(ref, hyp, debug = False):
    OP_OK, OP_SUB, OP_INS, OP_DEL = 0, 1, 2, 3
    DEL_PENALTY, INS_PENALTY, SUB_PENALTY = 1, 1, 1

    r = ref.split()
    h = hyp.split()

    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY
                insertionCost = costs[i][j-1] + INS_PENALTY
                deletionCost = costs[i-1][j] + DEL_PENALTY

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    i, j = len(r), len(h)
    numSub, numDel, numIns, numCor = 0, 0, 0, 0

    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("Ncor " + str(numCor))
        print("Nsub " + str(numSub))
        print("Ndel " + str(numDel))
        print("Nins " + str(numIns))

    wer_result = round((numSub + numDel + numIns) / (float) (len(r)), 3) * 100
    return {'WER' : wer_result, 'Cor' : numCor, 'Sub' : numSub, 'Ins' : numIns, 'Del' : numDel}