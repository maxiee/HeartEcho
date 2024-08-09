import 'dart:async';

import 'package:app/models/chat.dart';
import 'package:app/models/corpus.dart';
import 'package:app/models/training_session.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiClient {
  final String baseUrl;

  ApiClient({this.baseUrl = 'http://localhost:1127'});

  Future<String> chat(List<Map<String, dynamic>> history) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {'Content-Type': 'application/json', 'charset': 'utf-8'},
      body: jsonEncode({'history': history}),
    );
    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes))['response'];
    } else {
      throw Exception('Failed to load chat');
    }
  }

  Future<String> learn(
      List<ChatSession> chatSessions, List<String> knowledges) async {
    final response = await http.post(
      Uri.parse('$baseUrl/learn'),
      headers: {'Content-Type': 'application/json', 'charset': 'utf-8'},
      body: jsonEncode({
        'chat': chatSessions.map((e) => e.toHistory()).toList(),
        'knowledge': knowledges,
      }),
    );
    if (response.statusCode == 200) {
      return json.decode(utf8.decode(response.bodyBytes))['response'];
    } else {
      throw Exception('Failed to learn');
    }
  }

  Future<bool> saveModel() async {
    final response = await http.post(
      Uri.parse('$baseUrl/save_model'),
      headers: {'Content-Type': 'application/json'},
    );
    if (response.statusCode == 200) {
      return json.decode(response.body)['saved'];
    } else {
      throw Exception('Failed to save model');
    }
  }

  Future<List<dynamic>> fetchCorpora() async {
    final response = await http.get(Uri.parse('http://localhost:1127/corpus/'));
    if (response.statusCode == 200) {
      final List<dynamic> data =
          json.decode(utf8.decode(response.bodyBytes))['items'];
      return data;
    }
    return [];
  }

  Future<List<dynamic>> fetchCorpusEntries({
    required String corpusId,
    int skip = 0,
    int limit = 100,
  }) async {
    final queryParameters = {
      'corpus': corpusId,
      'skip': skip.toString(),
      'limit': limit.toString(),
    };

    final uri = Uri.parse('$baseUrl/corpus/entries')
        .replace(queryParameters: queryParameters);

    final response = await http.get(uri);
    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(utf8.decode(response.bodyBytes));
      return data.map((item) => CorpusEntry.fromJson(item)).toList();
    } else {
      throw Exception('Failed to fetch corpus entries: ${response.statusCode}');
    }
  }

  Future<Map<String, dynamic>> createCorpusEntry(
      Map<String, dynamic> data) async {
    final String corpusId =
        data['corpus_id']; // Assuming 'corpus_name' is actually the corpus ID
    final response = await http.post(
      Uri.parse('$baseUrl/corpus/entry?corpus_id=$corpusId'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'entry_type': data['entry_type'],
        'content': data['content'],
        'messages': data['messages'],
      }),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to create corpus entry: ${response.body}');
    }
  }

  Future<Map<String, dynamic>> updateCorpusEntry(
      String entryId, Map<String, dynamic> data) async {
    final response = await http.put(
      Uri.parse('$baseUrl/corpus_entry/$entryId'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode(data),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to update corpus entry: ${response.body}');
    }
  }

  Future<Map<String, dynamic>> createCorpus({
    required String name,
    required String description,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/corpus/'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'name': name, 'description': description}),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to create corpus: ${response.body}');
    }
  }

  Future<List<TrainingSession>> listSessions() async {
    final response = await http.get(
      Uri.parse('$baseUrl/sessions/list'),
    );
    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((json) => TrainingSession.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load sessions');
    }
  }

  Future<List<String>> getSavedModels() async {
    final response = await http.get(Uri.parse('$baseUrl/saved_models'));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return List<String>.from(data['models']);
    } else {
      throw Exception('Failed to load saved models');
    }
  }

  Future<TrainingSession> createNewTrainingSession(
      String name, String baseModel) async {
    final response = await http
        .post(
          Uri.parse('$baseUrl/sessions/'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({
            'name': name,
            'base_model': baseModel,
          }),
        )
        .timeout(const Duration(minutes: 30));

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return TrainingSession.fromJson(data);
    } else {
      throw Exception(
          'Failed to create new training session: ${response.body}');
    }
  }

  Future<TrainingSession> loadTrainingSession(String sessionId) async {
    final response = await http.post(
      Uri.parse('$baseUrl/sessions/load/$sessionId'),
      headers: {'Content-Type': 'application/json'},
    );
    if (response.statusCode == 200) {
      return TrainingSession.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load training session: ${response.body}');
    }
  }

  Future<TrainingSession?> getCurrentSession() async {
    final response = await http.get(Uri.parse('$baseUrl/sessions/current'));

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return TrainingSession.fromJson(data);
    } else if (response.statusCode == 404) {
      // 没有活跃的会话
      return null;
    } else {
      throw Exception('Failed to get current session');
    }
  }

  Future<TrainingSession> saveCurrentSession() async {
    final response = await http.post(
      Uri.parse('$baseUrl/sessions/save'),
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return TrainingSession.fromJson(data);
    } else {
      throw Exception('Failed to save current session: ${response.body}');
    }
  }

  Future<LossDistributionResponse> getErrorDistribution() async {
    final response =
        await http.get(Uri.parse('$baseUrl/corpus/loss_distribution'));
    if (response.statusCode == 200) {
      return LossDistributionResponse.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load error distribution');
    }
  }

  Future<int> getNewCorpusEntriesCount(String sessionName) async {
    final response = await http.get(
      Uri.parse(
          '$baseUrl/corpus/new_corpus_entries_count?session=$sessionName'),
    );
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return data['new_entries_count'];
    } else {
      throw Exception('Failed to load new corpus entries count');
    }
  }

  Future<Map<String, dynamic>> smeltNewCorpus(String sessionName) async {
    final response = await http.post(
      Uri.parse('$baseUrl/smelt_new_corpus'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'session': sessionName}),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to smelt new corpus: ${response.body}');
    }
  }
}

// ignore: non_constant_identifier_names
final API = ApiClient();
