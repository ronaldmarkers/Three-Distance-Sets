#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>  // std::pair
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/flags.h"
#include "ortools/base/init_google.h"
#include "ortools/base/logging.h"
#include "ortools/linear_solver/linear_solver.h"

using namespace std;

// Binomial Coefficient Program: https://stackoverflow.com/questions/55421835/c-binomial-coefficient-is-too-slow
int binomial(const int n, const int k) {
    std::vector<int> aSolutions(k);
    if (k == 0) return 1;  // Base case: binomial(n, 0) = 1
    if (k > n) return 0;  // Binomial coefficient is 0 when k > n

    aSolutions[0] = n - k + 1;

    for (int i = 1; i < k; ++i) {
        aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
    }

    return aSolutions[k - 1];
}

// Function to compute krav values
int krav(int k, int i, int n, int q) {
    int sum = 0;
    for (int j = 0; j <= k; j++) {
        sum += pow(-q, j) * pow(q - 1, k - j) * binomial(n - j, k - j) * binomial(i, j);
    }
    return sum;
}


vector<int> error = { 23 };
vector<int> hold;
vector<vector<int>> fail;
vector<double> values;
int q = 2;


namespace operations_research {
    double Maximize(vector<int> distances, int n) {
        // Create the linear solver with the GLOP backend.
        std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));

        const double infinity = solver->infinity();
        // x and y are non-negative variables.
        MPVariable* const var1 = solver->MakeNumVar(0.0, infinity, "var1");
        MPVariable* const var2 = solver->MakeNumVar(0.0, infinity, "var2");
        MPVariable* const var3 = solver->MakeNumVar(0.0, infinity, "var3");

        // Create the constraints, one per nutrient.
        std::vector<MPConstraint*> constraints;

        MPConstraint* const c0 = solver->MakeRowConstraint(0.0, infinity);
        c0->SetCoefficient(var1, 1);
        c0->SetCoefficient(var2, 0);
        c0->SetCoefficient(var3, 0);

        MPConstraint* const c1 = solver->MakeRowConstraint(0.0, infinity);
        c1->SetCoefficient(var1, 0);
        c1->SetCoefficient(var2, 1);
        c1->SetCoefficient(var3, 0);

        MPConstraint* const c2 = solver->MakeRowConstraint(0.0, infinity);
        c2->SetCoefficient(var1, 0);
        c2->SetCoefficient(var2, 0);
        c2->SetCoefficient(var3, 1);

        for (int p = 0; p <= n; p++) {

            std::vector<int> coeffs = { 0, 0, 0, 0 };

            for (int l = 0; l <= n; l++) {
                if (l == 0) {
                    coeffs[0] = krav(p, l, n, q);
                }
                if (l == distances[0]) {
                    coeffs[1] = krav(p, l, n, q);
                }
                if (l == distances[1]) {
                    coeffs[2] = krav(p, l, n, q);
                }
                if (l == distances[2]) {
                    coeffs[3] = krav(p, l, n, q);
                }
            }

            constraints.push_back(solver->MakeRowConstraint(-1 * coeffs[0], infinity));
            constraints.back()->SetCoefficient(var1, coeffs[1]);
            constraints.back()->SetCoefficient(var2, coeffs[2]);
            constraints.back()->SetCoefficient(var3, coeffs[3]);
        }

        MPObjective* const objective = solver->MutableObjective();
        objective->SetCoefficient(var1, 1);
        objective->SetCoefficient(var2, 1);
        objective->SetCoefficient(var3, 1);
        objective->SetMaximization();

        const MPSolver::ResultStatus result_status = solver->Solve();
        // Check that the problem has an optimal solution.
        if (result_status != MPSolver::OPTIMAL) {
            return 0;
        }

        return objective->Value();
    }
}

int main() {
    for (int n = 23; n <= 23; n++) {
        vector<vector<int>> index;
        int len = 0;
        vector<double> value;

        for (int i = 1; i <= n; i++) {
            for (int j = i + 1; j <= n; j++) {
                for (int k = j + 1; k <= n; k++) {
                    if ((i + j + k >= ((3 * n) / 2) + 1) || (-((n / 2) - i) * ((n / 2) - j) * ((n / 2) - k) + (n / 4) * (i + j + k) - (3 * pow(n, 2) / 8) > 0)) {
                        int Coe = 1 + n + (n * (n - 1)) / 2;
                        int U = floor(0.5 + sqrt(pow(Coe, 2) / (2 * Coe - 2) + 0.25));

                        if (n + (n * (n - 1) * (n - 2)) / 6 >= 2 * U) {
                            if (floor((double)j / (j - i) * k / (k - i)) == (double)j / (j - i) * k / (k - i)) {
                                if (floor((double)i / (i - j) * k / (k - j)) == (double)i / (i - j) * k / (k - j)) {
                                    if (floor((double)i / (i - k) * j / (j - k)) == (double)i / (i - k) * j / (j - k)) {
                                        if ((double)j / (j - i) * k / (k - i) <= U) {
                                            if ((double)i / (j - i) * k / (k - j) <= U) {
                                                if ((double)i / (k - i) * j / (k - j) <= U) {
                                                    index.push_back({ i, j, k });
                                                    len++;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            index.push_back({ i, j, k });
                            len++;
                        }
                    }
                }
            }
        }

        for (int ind = 0; ind < len; ind++) {
            double A;
            vector<int> dis = index[ind];
            cout << "{" << dis[0] << "," << dis[1] << "," << dis[2] << "}";
            A = operations_research::Maximize(dis, n);
            value.push_back(A);

            if(dis[0] == 8 && dis[1] == 12 && dis[2] == 16){
                cout << "{DUAL GALAY CODE VALUE: " << A << "}, ";
            }

            if (A == 0) {
                fail.push_back({ n, (int)A, dis[0], dis[1], dis[2] });
            }

            if (A >= n + (n * (n - 1) * (n - 2)) / 6) {
                fail.push_back({ n, (int)A, dis[0], dis[1], dis[2] });
            }
        }

        double G;

        if (!value.empty()) {
            double G = *max_element(value.begin(), value.end());
        }
        else {
            // Handle the case where the value vector is empty
            // For example, set G to a default value or throw an exception
            double G = 0; // or some other appropriate default value
        }

        values.push_back(G);
        if (G <= n + (n * (n - 1) * (n - 2)) / 6) {
            hold.push_back(n);
        }
    }

    cout << "Hold: ";
    for (int h : hold) cout << h << " ";
    cout << endl;

    cout << "Fail: " << endl;
    for (const auto& f : fail) {
        cout << "{";
        for (size_t i = 0; i < f.size(); i++) {
            cout << f[i];
            if (i < f.size() - 1) cout << ", ";
        }
        cout << "}, ";
    }
    cout << endl;

    return 0;
}
