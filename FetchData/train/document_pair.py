import json
import numpy as np
from sentence_transformers import SentenceTransformer # 384, 512, 768 chiều - không phụ thuộc kích thước của từ vựng
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
from typing import List, Dict, Tuple
from itertools import combinations

class CaDaoQueryGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Khởi tạo generator với model embedding
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        
    def load_documents(self, file_path: str = None, data: List[Dict] = None):
        """
        Load dữ liệu ca dao tục ngữ
        """
        if data:
            self.documents = data
        elif file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        
    def extract_keywords(self, text: str) -> List[str]:
        """
        Trích xuất từ khóa quan trọng
        """
        # Loại bỏ dấu câu và chuyển thành chữ thường
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Loại bỏ stop words tiếng Việt cơ bản
        stop_words = {'là', 'của', 'và', 'với', 'trong', 'trên', 'dưới', 'như', 'để', 'cho', 'về', 'có', 'được', 'này', 'đó', 'một', 'hai', 'ba', 'bốn', 'năm'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def calculate_relevance_scores(self, queries: List[str], documents: List[Dict]) -> List[Dict]:
        """
        Tính điểm relevance cho từng cặp query-document
        """
        results = []
        
        for query in queries:
            # Tạo embeddings
            query_embedding = self.model.encode([query])
            doc_texts = [doc['value'] for doc in documents]
            doc_embeddings = self.model.encode(doc_texts)
            
            # Tính cosine similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Tạo danh sách (doc_id, similarity_score)
            doc_similarities = [(i, sim) for i, sim in enumerate(similarities)]
            
            # Sắp xếp theo điểm số giảm dần
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Gán điểm relevance
            query_results = []
            for rank, (doc_idx, similarity) in enumerate(doc_similarities):
                if rank < 3:  # Top 1-3
                    relevance = 3
                elif rank < 7:  # Top 4-7
                    relevance = 2
                elif rank < 12:  # Top 8-12
                    relevance = 1
                else:  # Còn lại
                    relevance = 0
                
                result = {
                    "query": query,
                    "document_id": documents[doc_idx]['id'],
                    "document_value": documents[doc_idx]['value'],
                    "relevance": relevance,
                    "cosine_score": float(similarity),
                    "rank": rank + 1
                }
                query_results.append(result)
            
            results.extend(query_results)
        
        return results
    
    def generate_pairwise_training_data(self, queries: List[str], documents: List[Dict]) -> List[Dict]:
        """
        Tạo dữ liệu pairwise training cho RankNet
        Mỗi sample là một cặp (query, doc1, doc2, label)
        label = 1 nếu doc1 relevance > doc2, ngược lại = 0
        """
        pairwise_data = []
        
        for query in queries:
            # Tạo embeddings
            query_embedding = self.model.encode([query])
            doc_texts = [doc['value'] for doc in documents]
            doc_embeddings = self.model.encode(doc_texts)
            
            # Tính cosine similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Gán điểm relevance dựa trên similarity
            doc_relevances = []
            doc_similarities = [(i, sim) for i, sim in enumerate(similarities)]
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (doc_idx, similarity) in enumerate(doc_similarities):
                if rank < 3:  # Top 1-3
                    relevance = 3
                elif rank < 7:  # Top 4-7
                    relevance = 2
                elif rank < 12:  # Top 8-12
                    relevance = 1
                else:  # Còn lại
                    relevance = 0
                
                doc_relevances.append({
                    'doc_idx': doc_idx,
                    'relevance': relevance,
                    'similarity': similarity,
                    'document': documents[doc_idx]
                })
            
            # Tạo tất cả các cặp document có relevance khác nhau
            for i in range(len(doc_relevances)):
                for j in range(i + 1, len(doc_relevances)):
                    doc1 = doc_relevances[i]
                    doc2 = doc_relevances[j]
                    
                    # Gán nhãn theo thuật toán RankNet
                    if doc1['relevance'] > doc2['relevance']:
                        label = 1.0  # Doc1 tốt hơn Doc2
                    elif doc1['relevance'] < doc2['relevance']:
                        label = 0.0  # Doc1 kém hơn Doc2
                    else:
                        label = 0.5  # Doc1 và Doc2 tương đương (cùng relevance)
                    
                    pairwise_sample = {
                        "query": query,
                        "doc1_id": doc1['document']['id'],
                        "doc1_text": doc1['document']['value'],
                        "doc1_relevance": doc1['relevance'],
                        "doc1_similarity": float(doc1['similarity']),
                        "doc2_id": doc2['document']['id'],
                        "doc2_text": doc2['document']['value'],
                        "doc2_relevance": doc2['relevance'],
                        "doc2_similarity": float(doc2['similarity']),
                        "label": label  # 1.0, 0.5, hoặc 0.0
                    }
                    pairwise_data.append(pairwise_sample)
            
            # Tạo thêm một số cặp ngẫu nhiên để cân bằng dữ liệu
            # (Bỏ phần tạo cặp random vì đã có đủ cặp từ logic trên)
        
        return pairwise_data
        
        return pairwise_data
    
    def generate_vector_matrices(self, queries: List[str], documents: List[Dict]) -> Dict:
        """
        Tạo ma trận vector cho từng query và tất cả documents
        Format dễ đọc và xử lý cho việc training RankNet
        """
        vector_data = {
            "metadata": {
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "total_queries": len(queries),
                "total_documents": len(documents),
                "description": "Vector matrices for query-document pairs"
            },
            "queries": [],
            "documents_info": [],
            "matrices": []
        }
        
        # Lưu thông tin documents (chỉ cần 1 lần)
        for doc in documents:
            vector_data["documents_info"].append({
                "id": doc['id'],
                "text": doc['value'],
                "length": len(doc['value'])
            })
        
        # Tạo embeddings cho tất cả documents (chỉ cần tính 1 lần)
        doc_texts = [doc['value'] for doc in documents]
        all_doc_embeddings = self.model.encode(doc_texts)
        
        # Xử lý từng query
        for query_idx, query in enumerate(queries):
            print(f"Đang xử lý query {query_idx + 1}/{len(queries)}: {query}")
            
            # Tạo embedding cho query
            query_embedding = self.model.encode([query])[0]  # Lấy vector đầu tiên
            
            # Tính cosine similarity với tất cả documents
            similarities = cosine_similarity([query_embedding], all_doc_embeddings)[0]
            
            # Tạo ma trận cho query này
            query_matrix = {
                "query_id": query_idx,
                "query_text": query,
                "query_embedding": query_embedding.tolist(),  # Chuyển numpy array thành list
                "document_embeddings": all_doc_embeddings.tolist(),
                "similarity_scores": similarities.tolist(),
                "relevance_labels": [],
                "ranked_documents": []
            }
            
            # Gán relevance scores dựa trên similarity
            doc_similarities = [(i, sim) for i, sim in enumerate(similarities)]
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            
            relevance_labels = [0] * len(documents)  # Khởi tạo tất cả là 0
            
            for rank, (doc_idx, similarity) in enumerate(doc_similarities):
                if rank < 3:  # Top 1-3
                    relevance = 3
                elif rank < 7:  # Top 4-7
                    relevance = 2
                elif rank < 12:  # Top 8-12
                    relevance = 1
                else:  # Còn lại
                    relevance = 0
                
                relevance_labels[doc_idx] = relevance
                
                query_matrix["ranked_documents"].append({
                    "rank": rank + 1,
                    "document_id": documents[doc_idx]['id'],
                    "document_index": doc_idx,
                    "similarity_score": float(similarity),
                    "relevance": relevance
                })
            
            query_matrix["relevance_labels"] = relevance_labels
            
            # Lưu thông tin query
            vector_data["queries"].append({
                "id": query_idx,
                "text": query,
                "embedding_dimension": len(query_embedding)
            })
            
            # Lưu ma trận
            vector_data["matrices"].append(query_matrix)
        
        return vector_data
    
    def save_vector_matrices(self, vector_data: Dict, output_file: str):
        """
        Lưu ma trận vector ra file JSON với format dễ đọc
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)
        
        metadata = vector_data["metadata"]
        print(f"Đã lưu ma trận vector vào {output_file}")
        print(f"- Số queries: {metadata['total_queries']}")
        print(f"- Số documents: {metadata['total_documents']}")
        print(f"- Chiều embedding: {metadata['embedding_dimension']}")
        print(f"- Tổng số ma trận: {len(vector_data['matrices'])}")
    
    def save_results(self, results: List[Dict], output_file: str):
        """
        Lưu kết quả ra file JSON
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu {len(results)} training pairs vào {output_file}")
        
        # In thống kê
        if len(results) > 0 and 'label' in results[0]:
            # Thống kê cho pairwise data
            label_counts = {}
            relevance_diff_counts = {}
            
            for result in results:
                label = result['label']
                # Làm tròn label để nhóm (1.0, 0.5, 0.0)
                label_rounded = round(label, 1)
                label_counts[label_rounded] = label_counts.get(label_rounded, 0) + 1
                
                rel_diff = abs(result['doc1_relevance'] - result['doc2_relevance'])
                relevance_diff_counts[rel_diff] = relevance_diff_counts.get(rel_diff, 0) + 1
            
            print("Thống kê labels:")
            for label in sorted(label_counts.keys(), reverse=True):
                if label == 1.0:
                    print(f"  Label {label}: {label_counts[label]} pairs (Doc1 > Doc2)")
                elif label == 0.5:
                    print(f"  Label {label}: {label_counts[label]} pairs (Doc1 ≈ Doc2)")
                elif label == 0.0:
                    print(f"  Label {label}: {label_counts[label]} pairs (Doc1 < Doc2)")
                else:
                    print(f"  Label {label}: {label_counts[label]} pairs")
            
            print("Thống kê độ chênh lệch relevance:")
            for diff in sorted(relevance_diff_counts.keys()):
                print(f"  Chênh lệch {diff}: {relevance_diff_counts[diff]} pairs")
        else:
            # Thống kê cho relevance data cũ
            relevance_counts = {}
            for result in results:
                rel = result.get('relevance', 0)
                relevance_counts[rel] = relevance_counts.get(rel, 0) + 1
            
            print("Thống kê điểm relevance:")
            for rel in sorted(relevance_counts.keys(), reverse=True):
                print(f"  Relevance {rel}: {relevance_counts[rel]} pairs")

def load_documents_from_json(documents_path):
    try:
        with open(documents_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        documents = [item['value'] for item in data if 'value' in item and item['value'].strip()]
        if not documents:
            raise ValueError("No valid documents found in JSON file")
        return documents
    except Exception as e:
        print(f"Error loading documents from {documents_path}: {e}")
        raise

# Sử dụng
if __name__ == "__main__":
    # Dữ liệu mẫu
    documents_path = "cadao_tucngu_complete.json"
    data_path = "sample_01/input.json"
    output_path = "sample_01/output.json"

    with open(documents_path, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    # Khởi tạo generator
    generator = CaDaoQueryGenerator()
    generator.load_documents(data=sample_data)
    
    # TỰ NHẬP QUERY
    my_queries_01 = [
        "ý nghĩa tình anh em",
        "ca dao về quê hương",
        "tục ngữ về thầy cô",
        "tình cảm gia đình"
    ]
    my_queries_02 = [
        "Nhớ ơn công lao cha mẹ",
        "tấm gương hiếu thảo",
        "bài học cuộc sống",
        "câu nói hay về tình bạn"
    ]
    my_queries_03 = [
        "Nói về tình cảm quê hương",
        "Tình cảm gia đình bạn bè",
        "lời khuyên tuổi trẻ",
        "vẻ đẹp thiên nhiên Việt Nam",
        "những câu nói hay về cuộc sống"
    ]
    my_queries_04 = [
        "ca dao tục ngữ nói về tình cảm gia đình",
        "ý nghĩa của câu tục ngữ 'ăn quả nhớ kẻ trồng cây'",
        "tục ngữ phản ánh đạo lý uống nước nhớ nguồn",
        "câu ca dao nói về lòng yêu nước",
        "tìm hiểu giá trị giáo dục trong ca dao"
    ]
    my_queries_05 = [
        "tục ngữ về lòng kiên trì và bền bỉ",
        "ca dao tục ngữ nói về vai trò của học vấn",
        "tục ngữ dạy cách ứng xử trong cuộc sống",
        "ca dao nói về lòng nhân ái và sẻ chia"
    ]

    print("=== TẠO MA TRẬN VECTOR CHO BỘ DỮ LIỆU ===")
    
    # Tạo ma trận vector - Lưu ma trận vector vào input.json
    # vector_matrices = generator.generate_vector_matrices(my_queries_05, generator.documents)
    # generator.save_vector_matrices(vector_matrices, data_path)
    
    print("\n=== TẠO DỮ LIỆU PAIRWISE TRAINING CHO RANKNET ===")
    
    # Tạo dữ liệu pairwise training
    pairwise_dataset = generator.generate_pairwise_training_data(my_queries_01, generator.documents)
    
    # Lưu kết quả pairwise
    generator.save_results(pairwise_dataset, output_path)
    
    print("\n✅ Xong")
    # In một số ví dụ pairwise
    # print("\n=== VÍ DỤ CÁC PAIRWISE TRAINING SAMPLES ===")
    # for i, item in enumerate(pairwise_dataset[:5]):
    #     print(f"\nPairwise Sample {i+1}:")
    #     print(f"Query: {item['query']}")
    #     print(f"Doc1 (ID {item['doc1_id']}, Rel {item['doc1_relevance']}): {item['doc1_text'][:100]}...")
    #     print(f"Doc2 (ID {item['doc2_id']}, Rel {item['doc2_relevance']}): {item['doc2_text'][:100]}...")
    #     print(f"Label: {item['label']} ({'Doc1 > Doc2' if item['label'] == 1 else 'Doc1 < Doc2'})")
    #     print(f"Similarity scores: {item['doc1_similarity']:.4f} vs {item['doc2_similarity']:.4f}")
    
    # print(f"\n=== TỔNG KẾT ===")
    # print(f"Tổng số pairwise samples: {len(pairwise_dataset)}")
    
    # In ví dụ cách sử dụng ma trận vector
    # print("\n=== HƯỚNG DẪN SỬ DỤNG MA TRẬN VECTOR ===")
    # print("Để sử dụng ma trận vector từ input.json:")
    # print("1. Load file: data = json.load(open('sample_05/input.json'))")
    # print("2. Lấy ma trận query đầu tiên: matrix = data['matrices'][0]")
    # print("3. Lấy query embedding: query_vec = np.array(matrix['query_embedding'])")
    # print("4. Lấy tất cả doc embeddings: doc_vecs = np.array(matrix['document_embeddings'])")
    # print("5. Lấy relevance labels: labels = matrix['relevance_labels']")
    # print("6. Lấy similarity scores: similarities = matrix['similarity_scores']")