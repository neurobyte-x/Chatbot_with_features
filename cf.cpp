#include <bits/stdc++.h>
using namespace std;

int minSwaps(vector<int>& arr) {
    vector<int> sorted = arr;
    int n = arr.size();
    sort(sorted.begin(), sorted.end());
    vector<bool> vis(n, false);
    unordered_map<int, int> mp;
    int c_s = 0;
    int t_s = 0;

    for (int i = 0; i < sorted.size(); i++) {
        mp[sorted[i]] = i;
    }

    for (int i = 0; i < n; i++) {
        if (vis[i] == true || i == mp[arr[i]]) {
            continue;
        } else {
            int current = i;
            c_s = 0;
            while (vis[current] != true) {
                vis[current] = true;
                current = mp[arr[current]];
                c_s++;
            }
        }
        if (c_s > 0) {
            cout<<"cycle size: "<<c_s<<endl;
            t_s += c_s - 1;
            cout<<"total swaps so far: "<<t_s<<endl;
        }
    }
    return t_s;
}


int main() {
    
    vector<int> arr = {1, 3, 4, 5, 6};

    cout << "Minimum swaps required: " << minSwaps(arr) << endl;
    return 0;
}
