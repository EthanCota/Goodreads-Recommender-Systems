dataset = [(u, b, 1) for u, b, r in allRatings]
dataset_negative = []
for d in dataset[40000:75000]:
    selection = list(books - {b for b, _ in ratingsPerUser[d[0]]})

    dataset_negative.append((curr_user, random.sample(selection, 1)[0], 0))

dataset = dataset + dataset_negative
# Function to get predicted score for a user-book pair
def get_bpr_score(u_id, b_id):
    try:
        u_idx = userIDs[u_id]
        b_idx = itemIDs[b_id]
    except KeyError:
        return 0
    user_vec = user_factors[u_idx]
    item_vec = item_factors[b_idx]
    return np.dot(user_vec, item_vec)

def is_top_popular(b_id):
    return int(b_id in top_books)


def get_max_jaccard(u_id, b_id):
    users_b = set([u for u, _ in ratingsPerItem[b_id]])
    sims = []
    for b2, _ in ratingsPerUser[u_id]:
        users_b2 = set([u for u, _ in ratingsPerItem[b2]])
        curr_sim = Jaccard(users_b, users_b2)
        sims.append(curr_sim)
    return max(sims) if sims else 0

#BPR

#Re-Initializing user and item dictionaries
userIDs,itemIDs = {},{}
for d in dataset:
    u,i = d[0],d[1]
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)

nUsers,nItems = len(userIDs),len(itemIDs)

# dicts assparce matrix for B
Xui = scipy.sparse.lil_matrix((nUsers, nItems))
for d in dataset:
    Xui[userIDs[d[0]],itemIDs[d[1]]] = 1
Xui_csr = scipy.sparse.csr_matrix(Xui)

# Bayesian Personalized Ranking model with 5 latent factor
model = bpr.BayesianPersonalizedRanking(factors = 2)
model.fit(Xui_csr)

user_factors = model.user_factors  # Shape: (nUsers, factors)
item_factors = model.item_factors  # Shape: (nItems, factors)

book_popularity_rank = {item: rank for rank, (count, item) in enumerate(mostPopular)}

user_interaction_counts = {u: len(ratingsPerUser[u]) for u in userIDs}
book_interaction_counts = {b: len(ratingsPerItem[b]) for b in itemIDs}
# Model training
clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'AUC-ROC: {auc}')
print('Classification Report:')
print(report)


num_read = 0

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        # header
        predictions.write(l)
        continue
    u, b = l.strip().split(',')

    # Create feature vector for the current user-book pair
    features = {}
    features['bpr_score'] = get_bpr_score(u, b)
    features['is_top_popular'] = is_top_popular(b)
    features['popularity_rank'] = book_popularity_rank.get(b, len(book_popularity_rank))
    features['max_jaccard'] = get_max_jaccard(u, b)
    features['user_interactions'] = user_interaction_counts.get(u, 0)
    features['book_interactions'] = book_interaction_counts.get(b, 0)
    
    # Convert features to DataFrame and scale
    df_features = pd.DataFrame([features])
    df_features.fillna(0, inplace=True)
    X_scaled = scaler.transform(df_features)
    
    # Predict using logistic regression model
    prediction = clf.predict(X_scaled)[0]

    num_read += prediction
    
    predictions.write(u + ',' + b + ',' + str(prediction) + '\n')

predictions.close()


num_read