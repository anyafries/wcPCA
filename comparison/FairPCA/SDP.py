import pandas as pd
from FairPCA.utils import input_check

# for optimization
from cvxopt import matrix

# optimization part
import numpy as np
import cvxopt as cvx
import picos as pic

from numpy import array


# use log to avoid the case product can overflow the floating point representation
def geo_mean_through_log(numberList):
    # if some is 0, return 0.
    if np.amin(numberList) <= 1.0e-12:
        return 0

    logNumberList = np.log(numberList)
    return np.exp(logNumberList.sum() / len(numberList))


# the SDP for fair PCA method


def fairDimReductionFractional(
    n,
    k,
    B,
    list_n="one",
    Obj="MM_Loss",
    list_d="all",
    verbose=1,
    print_other_obj=True,
    return_option="run_statistics",
    save=True,
    savedPath="fairDimReductionFractional.csv",
):
    """
    Given k PSD n-by-n matrices B1,...,Bk, solve the (fractional) convex 
    optimization of fair dimensional reduction.
    Arguments:
        n:  original number of features (size of all B_i's)
        k:  number of groups
        B:  list of PSD matrices, as numpy matrices. 
            It must contain at least k matrices.
            If there are more than k matrices provided, the first k will be 
            used as k groups.
        list_n: by default, this is simply n, which is the total number of 
            features. If 'all', this n0 (number of dimension, first n0 features 
            are used) ranges from d0+1 to n (d0 is the target dimension of that 
            iteration). Else, you can specify as a list of n_0.
        list_d: list of target dimensions to project to. By default ('all'), 
            it is from 1 to n-1.
        print_other_obj: setting to True will also print other welfare economic 
            objective (total of four, including one specified as input Obj)
        verbose: set to 1 if the details are to be printed. 
            Set to 2 to print the information table of each iteration
        save: will save to csv if set to True
        savedPath: path of the file to export the result to.
        Obj: the objective to optimize. 
            - MM_Var (maximize the minimum variance)
            - MM_Loss (default) (minimize the maximum loss, output the negative)
            - NSW (Nash social welfare)
        return_option:  
            "run_statistics": (default) returns the runtime, n,d, the rank, and 
                several objectives of each group.
            'frac_sol': return a list of fractional solution X.
                - list_X: a list where each entry is [n,d,X], where X is the
                  solution matrix X as cvx matrix. 
                  One can convert X back to numpy by:
                    import numpy as np
                    array(list_X[0][2]) 
                    # gives np solution matrix X of the first setting of (n,d).
                    array(list_X[i][2]) 
                    # gives np solution matrix X of the (i+1)th setting of (n,d).
    """

    # input check
    if input_check(n, k, 1, B, function_name="fairDimReductionFractional") > 0:
        return -1

    # for storing results of the optimization
    runstats = []
    if return_option == "frac_sol":
        list_X = []

    # list of all d
    if list_d == "all":
        list_d = range(1, n)

    for d in list_d:
        # valid value of n_0
        if list_n == "one":
            list_n_this_d = [n]
        elif list_n == "all":
            list_n_this_d = range(d + 1, n + 1)
        else:
            list_n_this_d = list_n

        for n0 in list_n_this_d:
            # shorten version of the matrix, in case we want to delete any earlier features for experiments
            Bnumpy_s = [B[i][np.ix_(range(0, n0), range(0, n0))] for i in range(k)]

            # now define the problem
            B_s = [matrix(B[i][np.ix_(range(0, n0), range(0, n0))]) for i in range(k)]

            fairPCA = pic.Problem()
            n = n0

            I = pic.Constant(
                "I", cvx.spmatrix([1] * n, range(n), range(n), (n, n))
            )  # identity matrix

            # Add the symmetric matrix variable.
            X = fairPCA.add_variable(
                "X", (n, n), "symmetric"
            )  # projection matrix, should be rank d but relaxed
            z = fairPCA.add_variable("z", 1)  # scalar, for the objective

            # Add parameters for each group
            A = [pic.Constant("A" + str(i), B_s[i]) for i in range(k)]

            # best possible variance for each group
            best = [
                np.sum(np.sort(np.linalg.eigvalsh(Bnumpy_s[i]))[-d:]) for i in range(k)
            ]

            # Constrain X on trace
            fairPCA.add_constraint(I | X <= d)

            # Constrain X to be positive semidefinite.
            fairPCA.add_constraint(X >> 0)
            fairPCA.add_constraint(X << I)

            # the following depends on the type of the problems. Here we coded 3 of them:
            # 1) max min variance 2) min max loss 3) Nash social welfare of variance

            if Obj == "MM_Loss":
                # Add loss constriant
                fairPCA.add_list_of_constraints(
                    [(A[i] | X) - best[i] >= z for i in range(k)]
                )  # constraints

                # Set the objective.
                fairPCA.set_objective("max", z)

            elif Obj == "MM_Var":
                # Add variance constriant
                fairPCA.add_list_of_constraints(
                    [(A[i] | X) >= z for i in range(k)]
                )  # constraints

                # Set the objective.
                fairPCA.set_objective("max", z)

            elif Obj == "NSW":
                s = fairPCA.add_variable("s", k)  # vector of variances
                # Add variance constriant
                fairPCA.add_list_of_constraints(
                    [(A[i] | X) >= s[i] for i in range(k)]
                )  # constraints

                # Set the objective.
                fairPCA.add_constraint(z <= pic.geomean(s))
                fairPCA.set_objective("max", z)

            else:
                fairPCA.set_objective("max", z)
                print(
                    "Error: fairDimReductionFractional is called with invalid Objective. Supported Obj augements are: ... Exit the method"
                )
                return

            solveInfo = fairPCA.solve(verbosity=0, solver="cvxopt")

            var = [np.sum(np.multiply(Bnumpy_s[i], X.value)) for i in range(k)]
            loss = [var[i] - best[i] for i in range(k)]

            # find rank of X
            rank = np.linalg.matrix_rank(X.value)

            # dictionary of info for this iterate
            # solveInfoShort = dict((key, solveInfo[key]) for key in ('time','obj','status'))
            solveInfoShort = {
                "d": d,
                "rank": rank,
                "time": solveInfo.searchTime,
                "obj": solveInfo.value,
                "status": solveInfo.status,
            }

            # add information of this optimization for this d,n0
            runstats.append(pd.DataFrame(solveInfoShort, index=[n0]))  # add this info

            if return_option == "frac_sol":
                list_X.append([n0, d, X.value])

    runstats = pd.concat(runstats)

    if verbose == 2:
        print(runstats)

    if verbose == 1:
        print("The total number of cases tested is:")
        print(len(runstats))
        print("The number of cases where the rank is exact is:")
        print(len(runstats[runstats["d"] == runstats["rank"]]))

    if save:
        runstats.to_csv(savedPath, index=False)

    if return_option == "frac_sol":
        return (runstats, list_X)

    return runstats
