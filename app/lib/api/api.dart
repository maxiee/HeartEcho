import 'package:app/models/chat.dart';
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

  Future<List<dynamic>?> fetchCorpora() async {
    final response = await http.get(Uri.parse('http://localhost:1127/corpus'));
    if (response.statusCode == 200) {
      final List<dynamic> data =
          json.decode(json.decode(response.body)['corpora']);
      return data;
    }
    return null;
  }

  Future<List<dynamic>?> fetchCorpusEntries(String corpusId) async {
    final response = await http.get(
        Uri.parse('http://localhost:1127/corpus/$corpusId/corpus_entries'));
    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body)['entries'];
      return data;
    }
    return null;
  }
}

// ignore: non_constant_identifier_names
final API = ApiClient();
