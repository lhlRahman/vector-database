// Copyright [year] <Owner>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../features/atomic_persistence.hpp"      // for PersistenceConfig + to_json
#include "../features/recovery_state_machine.hpp"  // for RecoveryInfo + to_json
#include "../core/vector_database.hpp"
#include "vector_db_server.hpp"
#include "../json.hpp"  // your single-header JSON

using json = nlohmann::json;

VectorDBServer::VectorDBServer(size_t dims,
                               const std::string& dbFile,
                               const std::string& host,
                               int port,
                               bool enable_recovery,
                               bool enable_batch,
                               bool enable_stats)
    : dimensions(dims),
      db_file(dbFile),
      host(host),
      port(port),
      enable_recovery_endpoints(enable_recovery),
      enable_batch_endpoints(enable_batch),
      enable_statistics_endpoints(enable_stats) {
  // Configure persistence
  PersistenceConfig persistence_config;
  persistence_config.data_directory          = "data";
  persistence_config.log_directory           = "logs";
  persistence_config.checkpoint_interval     = std::chrono::minutes(5);
  persistence_config.checkpoint_trigger_ops  = 1000;
  persistence_config.log_rotation_size       = 100 * 1024 * 1024; // 100MB
  persistence_config.max_log_files           = 10;

  // Build database (exact search, atomic persistence on, batch ops on)
  db = std::make_unique<VectorDatabase>(dimensions, "exact", /*enable_atomic_persistence=*/true,
                                        /*enable_batch_operations=*/true, persistence_config);

  setupRoutes();
  startRecoveryMonitoring();
}

VectorDBServer::~VectorDBServer() {
  stopRecoveryMonitoring();
  if (db) {
    db->shutdown();
  }
}

void VectorDBServer::setupRoutes() {
  setupHealthRoutes();
  setupVectorRoutes();

  if (enable_batch_endpoints) {
    setupBatchRoutes();
  }
  if (enable_recovery_endpoints) {
    setupRecoveryRoutes();
  }
  if (enable_statistics_endpoints) {
    setupStatisticsRoutes();
  }
  setupConfigRoutes();
}

void VectorDBServer::setupHealthRoutes() {
  server.Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
    handleHealth(req, res);
  });
}

void VectorDBServer::setupVectorRoutes() {
  server.Get("/vectors", [this](const httplib::Request& req, httplib::Response& res) {
    handleGetVectors(req, res);
  });

  server.Post("/vectors", [this](const httplib::Request& req, httplib::Response& res) {
    handlePostVectors(req, res);
  });

  server.Get(R"(/vectors/(.*))", [this](const httplib::Request& req, httplib::Response& res) {
    handleGetVector(req, res);
  });

  server.Put(R"(/vectors/(.*))", [this](const httplib::Request& req, httplib::Response& res) {
    handlePutVector(req, res);
  });

  server.Delete(R"(/vectors/(.*))", [this](const httplib::Request& req, httplib::Response& res) {
    handleDeleteVector(req, res);
  });

  server.Post("/search", [this](const httplib::Request& req, httplib::Response& res) {
    handleSearch(req, res);
  });

  server.Post("/search/batch", [this](const httplib::Request& req, httplib::Response& res) {
    handleBatchSearch(req, res);
  });
}

void VectorDBServer::setupBatchRoutes() {
  server.Post("/vectors/batch/insert", [this](const httplib::Request& req, httplib::Response& res) {
    handleBatchInsert(req, res);
  });

  server.Put("/vectors/batch/update", [this](const httplib::Request& req, httplib::Response& res) {
    handleBatchUpdate(req, res);
  });

  server.Delete("/vectors/batch/delete", [this](const httplib::Request& req, httplib::Response& res) {
    handleBatchDelete(req, res);
  });

  server.Post("/vectors/batch/mixed", [this](const httplib::Request& req, httplib::Response& res) {
    handleMixedBatch(req, res);
  });
}

void VectorDBServer::setupRecoveryRoutes() {
  server.Get("/recovery/status", [this](const httplib::Request& req, httplib::Response& res) {
    handleRecoveryStatus(req, res);
  });

  server.Get("/recovery/info", [this](const httplib::Request& req, httplib::Response& res) {
    handleRecoveryInfo(req, res);
  });

  server.Post("/recovery/checkpoint", [this](const httplib::Request& req, httplib::Response& res) {
    handleForceCheckpoint(req, res);
  });

  server.Post("/recovery/flush", [this](const httplib::Request& req, httplib::Response& res) {
    handleForceFlush(req, res);
  });
}

void VectorDBServer::setupStatisticsRoutes() {
  server.Get("/statistics", [this](const httplib::Request& req, httplib::Response& res) {
    handleStatistics(req, res);
  });

  server.Get("/statistics/database", [this](const httplib::Request& req, httplib::Response& res) {
    handleDatabaseStats(req, res);
  });

  server.Get("/statistics/persistence", [this](const httplib::Request& req, httplib::Response& res) {
    handlePersistenceStats(req, res);
  });

  server.Get("/statistics/batch", [this](const httplib::Request& req, httplib::Response& res) {
    handleBatchStats(req, res);
  });
}

void VectorDBServer::setupConfigRoutes() {
  server.Get("/config", [this](const httplib::Request& req, httplib::Response& res) {
    handleGetConfig(req, res);
  });

  server.Put("/config", [this](const httplib::Request& req, httplib::Response& res) {
    handleUpdateConfig(req, res);
  });

  server.Put("/config/persistence", [this](const httplib::Request& req, httplib::Response& res) {
    handleUpdatePersistenceConfig(req, res);
  });
}

void VectorDBServer::handleHealth(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;

  json response;
  response["status"]                = "healthy";
  response["database_ready"]        = db->isReady();
  response["recovery_in_progress"]  = db->isRecovering();
  response["dimensions"]            = dimensions;
  response["total_vectors"]         = db->getAllVectors().size();
  response["timestamp"]             = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch()).count();

  res.set_content(response.dump(), "application/json");
  successful_requests++;
  logRequest("GET", "/health", 200);
}

void VectorDBServer::handleGetVectors(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;
  try {
    std::lock_guard<std::mutex> lock(db_mutex);
    json response;
    response["vectors"] = json::array();

    for (const auto& [key, vector] : db->getAllVectors()) {
      json v;
      v["key"]    = key;
      v["vector"] = std::vector<float>(vector.begin(), vector.end());
      auto m      = db->getMetadata(key);
      if (!m.empty()) v["metadata"] = m;
      response["vectors"].push_back(v);
    }

    response["count"]      = db->getAllVectors().size();
    response["dimensions"] = dimensions;

    res.set_content(response.dump(), "application/json");
    successful_requests++;
    logRequest("GET", "/vectors", 200);
  } catch (const std::exception& e) {
    handleError(res, 500, std::string("Internal server error: ") + e.what());
    failed_requests++;
    logRequest("GET", "/vectors", 500);
  }
}

void VectorDBServer::handlePostVectors(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  try {
    auto body = json::parse(req.body);
    if (!body.contains("key") || !body.contains("vector")) {
      handleError(res, 400, "Missing required fields: key, vector");
      failed_requests++;
      logRequest("POST", "/vectors", 400);
      return;
    }

    std::string key      = body["key"];
    std::string metadata = body.value("metadata", "");

    if (!validateVector(body["vector"], dimensions)) {
      handleError(res, 400, "Invalid vector format or dimensions");
      failed_requests++;
      logRequest("POST", "/vectors", 400);
      return;
    }

    std::vector<float> data = body["vector"];
    Vector v(data);

    std::lock_guard<std::mutex> lock(db_mutex);
    if (db->insert(v, key, metadata)) {
      json r;
      r["success"] = true;
      r["key"]     = key;
      r["message"] = "Vector inserted successfully";
      res.set_content(r.dump(), "application/json");
      successful_requests++;
      logRequest("POST", "/vectors", 200);
    } else {
      handleError(res, 409, "Vector with key already exists or insertion failed");
      failed_requests++;
      logRequest("POST", "/vectors", 409);
    }
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("POST", "/vectors", 400);
  }
}

void VectorDBServer::handleGetVector(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  try {
    std::string key = req.matches[1];
    std::lock_guard<std::mutex> lock(db_mutex);

    const Vector* vec = db->get(key);
    if (vec) {
      json r;
      r["key"]    = key;
      r["vector"] = std::vector<float>(vec->begin(), vec->end());
      auto m      = db->getMetadata(key);
      if (!m.empty()) r["metadata"] = m;
      res.set_content(r.dump(), "application/json");
      successful_requests++;
      logRequest("GET", "/vectors/" + key, 200);
    } else {
      handleError(res, 404, "Vector not found");
      failed_requests++;
      logRequest("GET", "/vectors/" + key, 404);
    }
  } catch (const std::exception& e) {
    handleError(res, 500, std::string("Internal server error: ") + e.what());
    failed_requests++;
    logRequest("GET", "/vectors/*", 500);
  }
}

void VectorDBServer::handlePutVector(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  try {
    std::string key = req.matches[1];
    auto body = json::parse(req.body);

    if (!body.contains("vector")) {
      handleError(res, 400, "Missing required field: vector");
      failed_requests++;
      logRequest("PUT", "/vectors/" + key, 400);
      return;
    }

    std::string metadata = body.value("metadata", "");

    if (!validateVector(body["vector"], dimensions)) {
      handleError(res, 400, "Invalid vector format or dimensions");
      failed_requests++;
      logRequest("PUT", "/vectors/" + key, 400);
      return;
    }

    std::vector<float> data = body["vector"];
    Vector v(data);

    std::lock_guard<std::mutex> lock(db_mutex);
    if (db->update(v, key, metadata)) {
      json r;
      r["success"] = true;
      r["key"]     = key;
      r["message"] = "Vector updated successfully";
      res.set_content(r.dump(), "application/json");
      successful_requests++;
      logRequest("PUT", "/vectors/" + key, 200);
    } else {
      handleError(res, 404, "Vector not found or update failed");
      failed_requests++;
      logRequest("PUT", "/vectors/" + key, 404);
    }
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("PUT", "/vectors/*", 400);
  }
}

void VectorDBServer::handleDeleteVector(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  try {
    std::string key = req.matches[1];
    std::lock_guard<std::mutex> lock(db_mutex);

    if (db->remove(key)) {
      json r;
      r["success"] = true;
      r["key"]     = key;
      r["message"] = "Vector deleted successfully";
      res.set_content(r.dump(), "application/json");
      successful_requests++;
      logRequest("DELETE", "/vectors/" + key, 200);
    } else {
      handleError(res, 404, "Vector not found");
      failed_requests++;
      logRequest("DELETE", "/vectors/" + key, 404);
    }
  } catch (const std::exception& e) {
    handleError(res, 500, std::string("Internal server error: ") + e.what());
    failed_requests++;
    logRequest("DELETE", "/vectors/*", 500);
  }
}

void VectorDBServer::handleSearch(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  try {
    auto body = json::parse(req.body);
    if (!body.contains("query") || !body.contains("k")) {
      handleError(res, 400, "Missing required fields: query, k");
      failed_requests++;
      logRequest("POST", "/search", 400);
      return;
    }
    if (!validateVector(body["query"], dimensions)) {
      handleError(res, 400, "Invalid query vector format or dimensions");
      failed_requests++;
      logRequest("POST", "/search", 400);
      return;
    }

    std::vector<float> q = body["query"];
    Vector query(q);
    size_t k = body["k"];
    bool include_metadata = body.value("include_metadata", false);

    std::lock_guard<std::mutex> lock(db_mutex);

    json response;
    response["query"]   = q;
    response["k"]       = k;
    response["results"] = json::array();

    if (include_metadata) {
      auto results = db->similaritySearchWithMetadata(query, k);
      for (const auto& r : results) {
        json item;
        item["key"]       = r.key;
        item["distance"]  = r.distance;
        item["metadata"]  = r.metadata;
        response["results"].push_back(item);
      }
    } else {
      auto results = db->similaritySearch(query, k);
      for (const auto& [key, dist] : results) {
        json item;
        item["key"]      = key;
        item["distance"] = dist;
        response["results"].push_back(item);
      }
    }
    response["count"] = response["results"].size();

    res.set_content(response.dump(), "application/json");
    successful_requests++;
    logRequest("POST", "/search", 200);
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("POST", "/search", 400);
  }
}

void VectorDBServer::handleBatchSearch(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  try {
    auto body = json::parse(req.body);
    if (!body.contains("queries") || !body.contains("k")) {
      handleError(res, 400, "Missing required fields: queries, k");
      failed_requests++;
      logRequest("POST", "/search/batch", 400);
      return;
    }

    std::vector<Vector> queries;
    for (const auto& qj : body["queries"]) {
      if (!validateVector(qj, dimensions)) {
        handleError(res, 400, "Invalid query vector format or dimensions");
        failed_requests++;
        logRequest("POST", "/search/batch", 400);
        return;
      }
      std::vector<float> q = qj;
      queries.emplace_back(q);
    }

    size_t k = body["k"];
    std::lock_guard<std::mutex> lock(db_mutex);

    auto results = db->batchSimilaritySearch(queries, k);
    json response;
    response["queries"]  = body["queries"];
    response["k"]        = k;
    response["results"]  = json::array();

    for (const auto& per_query : results) {
      json arr = json::array();
      for (const auto& [key, dist] : per_query) {
        json item;
        item["key"]      = key;
        item["distance"] = dist;
        arr.push_back(item);
      }
      response["results"].push_back(arr);
    }
    response["query_count"] = queries.size();

    res.set_content(response.dump(), "application/json");
    successful_requests++;
    logRequest("POST", "/search/batch", 200);
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("POST", "/search/batch", 400);
  }
}

void VectorDBServer::handleBatchInsert(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  total_batch_operations++;
  try {
    auto body = json::parse(req.body);
    if (!body.contains("keys") || !body.contains("vectors")) {
      handleError(res, 400, "Missing required fields: keys, vectors");
      failed_requests++;
      logRequest("POST", "/vectors/batch/insert", 400);
      return;
    }
    if (!validateBatchRequest(body)) {
      handleError(res, 400, "Invalid batch request format");
      failed_requests++;
      logRequest("POST", "/vectors/batch/insert", 400);
      return;
    }

    std::vector<std::string> keys     = body["keys"];
    std::vector<std::string> metadata = body.value("metadata", std::vector<std::string>{});
    std::vector<Vector> vectors;

    for (const auto& vj : body["vectors"]) {
      std::vector<float> vf = vj;
      vectors.emplace_back(vf);
    }

    std::lock_guard<std::mutex> lock(db_mutex);
    auto batch_result = db->batchInsert(keys, vectors, metadata);

    json response;
    response["success"]               = batch_result.success;
    response["operations_committed"]  = batch_result.operations_committed;
    response["transaction_id"]        = batch_result.transaction_id;
    response["duration_ms"]           = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          batch_result.duration).count();

    if (!batch_result.success) {
      response["error_message"] = batch_result.error_message;
      res.status = 500;
      failed_requests++;
      logRequest("POST", "/vectors/batch/insert", 500);
    } else {
      successful_requests++;
      logRequest("POST", "/vectors/batch/insert", 200);
    }
    res.set_content(response.dump(), "application/json");
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("POST", "/vectors/batch/insert", 400);
  }
}

void VectorDBServer::handleBatchUpdate(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  total_batch_operations++;
  try {
    auto body = json::parse(req.body);
    if (!body.contains("keys") || !body.contains("vectors")) {
      handleError(res, 400, "Missing required fields: keys, vectors");
      failed_requests++;
      logRequest("PUT", "/vectors/batch/update", 400);
      return;
    }
    if (!validateBatchRequest(body)) {
      handleError(res, 400, "Invalid batch request format");
      failed_requests++;
      logRequest("PUT", "/vectors/batch/update", 400);
      return;
    }

    std::vector<std::string> keys     = body["keys"];
    std::vector<std::string> metadata = body.value("metadata", std::vector<std::string>{});
    std::vector<Vector> vectors;

    for (const auto& vj : body["vectors"]) {
      std::vector<float> vf = vj;
      vectors.emplace_back(vf);
    }

    std::lock_guard<std::mutex> lock(db_mutex);
    auto batch_result = db->batchUpdate(keys, vectors, metadata);

    json response;
    response["success"]               = batch_result.success;
    response["operations_committed"]  = batch_result.operations_committed;
    response["transaction_id"]        = batch_result.transaction_id;
    response["duration_ms"]           = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          batch_result.duration).count();

    if (!batch_result.success) {
      response["error_message"] = batch_result.error_message;
      res.status = 500;
      failed_requests++;
      logRequest("PUT", "/vectors/batch/update", 500);
    } else {
      successful_requests++;
      logRequest("PUT", "/vectors/batch/update", 200);
    }
    res.set_content(response.dump(), "application/json");
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("PUT", "/vectors/batch/update", 400);
  }
}

void VectorDBServer::handleBatchDelete(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  total_batch_operations++;
  try {
    auto body = json::parse(req.body);
    if (!body.contains("keys")) {
      handleError(res, 400, "Missing required field: keys");
      failed_requests++;
      logRequest("DELETE", "/vectors/batch/delete", 400);
      return;
    }

    std::vector<std::string> keys = body["keys"];
    std::lock_guard<std::mutex> lock(db_mutex);

    auto batch_result = db->batchDelete(keys);

    json response;
    response["success"]               = batch_result.success;
    response["operations_committed"]  = batch_result.operations_committed;
    response["transaction_id"]        = batch_result.transaction_id;
    response["duration_ms"]           = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          batch_result.duration).count();

    if (!batch_result.success) {
      response["error_message"] = batch_result.error_message;
      res.status = 500;
      failed_requests++;
      logRequest("DELETE", "/vectors/batch/delete", 500);
    } else {
      successful_requests++;
      logRequest("DELETE", "/vectors/batch/delete", 200);
    }
    res.set_content(response.dump(), "application/json");
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("DELETE", "/vectors/batch/delete", 400);
  }
}

void VectorDBServer::handleMixedBatch(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  total_batch_operations++;
  try {
    auto body = json::parse(req.body);
    if (!body.contains("operations")) {
      handleError(res, 400, "Missing required field: operations");
      failed_requests++;
      logRequest("POST", "/vectors/batch/mixed", 400);
      return;
    }
    // Not implemented
    handleError(res, 501, "Mixed batch operations not yet implemented");
    failed_requests++;
    logRequest("POST", "/vectors/batch/mixed", 501);
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("POST", "/vectors/batch/mixed", 400);
  }
}

void VectorDBServer::handleRecoveryStatus(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;

  json response;
  response["recovery_in_progress"] = db->isRecovering();
  response["database_ready"]       = db->isReady();
  response["recovery_info"]        = db->getRecoveryInfo(); // requires to_json(RecoveryInfo)

  res.set_content(response.dump(), "application/json");
  successful_requests++;
  logRequest("GET", "/recovery/status", 200);
}

void VectorDBServer::handleRecoveryInfo(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;

  json response;
  response["recovery_info"] = db->getRecoveryInfo(); // requires to_json(RecoveryInfo)

  res.set_content(response.dump(), "application/json");
  successful_requests++;
  logRequest("GET", "/recovery/info", 200);
}

void VectorDBServer::handleForceCheckpoint(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;
  try {
    std::lock_guard<std::mutex> lock(db_mutex);
    if (db->checkpoint()) {
      json r;
      r["success"] = true;
      r["message"] = "Checkpoint created successfully";
      res.set_content(r.dump(), "application/json");
      successful_requests++;
      logRequest("POST", "/recovery/checkpoint", 200);
    } else {
      handleError(res, 500, "Failed to create checkpoint");
      failed_requests++;
      logRequest("POST", "/recovery/checkpoint", 500);
    }
  } catch (const std::exception& e) {
    handleError(res, 500, std::string("Internal server error: ") + e.what());
    failed_requests++;
    logRequest("POST", "/recovery/checkpoint", 500);
  }
}

void VectorDBServer::handleForceFlush(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;
  try {
    std::lock_guard<std::mutex> lock(db_mutex);
    size_t ops = db->flush();

    json r;
    r["success"]            = true;
    r["operations_flushed"] = ops;
    r["message"]            = "Flush completed successfully";
    res.set_content(r.dump(), "application/json");

    successful_requests++;
    logRequest("POST", "/recovery/flush", 200);
  } catch (const std::exception& e) {
    handleError(res, 500, std::string("Internal server error: ") + e.what());
    failed_requests++;
    logRequest("POST", "/recovery/flush", 500);
  }
}

void VectorDBServer::handleStatistics(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;

  auto server_stats = getServerStatistics();
  auto db_stats     = db->getStatistics();

  json response;
  response["server"] = {
      {"total_requests", server_stats.total_requests},
      {"successful_requests", server_stats.successful_requests},
      {"failed_requests", server_stats.failed_requests},
      {"total_batch_operations", server_stats.total_batch_operations},
      {"recovery_in_progress", server_stats.recovery_in_progress},
      {"server_running", server_stats.server_running},
      {"host", server_stats.host},
      {"port", server_stats.port}
  };

  response["database"] = {
      {"total_vectors", db_stats.total_vectors},
      {"total_inserts", db_stats.total_inserts},
      {"total_searches", db_stats.total_searches},
      {"total_updates", db_stats.total_updates},
      {"total_deletes", db_stats.total_deletes},
      {"dimensions", db_stats.dimensions},
      {"algorithm", db_stats.algorithm},
      {"atomic_persistence_enabled", db_stats.atomic_persistence_enabled},
      {"batch_operations_enabled", db_stats.batch_operations_enabled}
  };

  res.set_content(response.dump(), "application/json");
  successful_requests++;
  logRequest("GET", "/statistics", 200);
}

void VectorDBServer::handleDatabaseStats(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;

  auto db_stats = db->getStatistics();

  json response;
  response["total_vectors"]               = db_stats.total_vectors;
  response["total_inserts"]               = db_stats.total_inserts;
  response["total_searches"]              = db_stats.total_searches;
  response["total_updates"]               = db_stats.total_updates;
  response["total_deletes"]               = db_stats.total_deletes;
  response["dimensions"]                  = db_stats.dimensions;
  response["algorithm"]                   = db_stats.algorithm;
  response["atomic_persistence_enabled"]  = db_stats.atomic_persistence_enabled;
  response["batch_operations_enabled"]    = db_stats.batch_operations_enabled;

  res.set_content(response.dump(), "application/json");
  successful_requests++;
  logRequest("GET", "/statistics/database", 200);
}

void VectorDBServer::handlePersistenceStats(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;

  auto db_stats = db->getStatistics();
  json response;
  response["persistence_stats"] = db_stats.persistence_stats; // relies on to_json for that struct if you want nested JSON

  res.set_content(response.dump(), "application/json");
  successful_requests++;
  logRequest("GET", "/statistics/persistence", 200);
}

void VectorDBServer::handleBatchStats(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;

  auto db_stats = db->getStatistics();
  json response;
  response["batch_stats"] = db_stats.batch_stats;

  res.set_content(response.dump(), "application/json");
  successful_requests++;
  logRequest("GET", "/statistics/batch", 200);
}

void VectorDBServer::handleGetConfig(const httplib::Request& /*req*/, httplib::Response& res) {
  total_requests++;

  json response;
  response["dimensions"]                    = dimensions;
  response["host"]                          = host;
  response["port"]                          = port;
  response["enable_recovery_endpoints"]     = enable_recovery_endpoints;
  response["enable_batch_endpoints"]        = enable_batch_endpoints;
  response["enable_statistics_endpoints"]   = enable_statistics_endpoints;

  // Serialize your PersistenceConfig via to_json
  response["persistence_config"] = db->getPersistenceConfig();

  res.set_content(response.dump(), "application/json");
  successful_requests++;
  logRequest("GET", "/config", 200);
}

void VectorDBServer::handleUpdateConfig(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  try {
    auto body = json::parse(req.body);

    if (body.contains("host"))  host = body["host"];
    if (body.contains("port"))  port = body["port"];
    if (body.contains("enable_recovery_endpoints"))   enable_recovery_endpoints   = body["enable_recovery_endpoints"];
    if (body.contains("enable_batch_endpoints"))      enable_batch_endpoints      = body["enable_batch_endpoints"];
    if (body.contains("enable_statistics_endpoints")) enable_statistics_endpoints = body["enable_statistics_endpoints"];

    json r;
    r["success"] = true;
    r["message"] = "Configuration updated successfully";
    res.set_content(r.dump(), "application/json");
    successful_requests++;
    logRequest("PUT", "/config", 200);
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("PUT", "/config", 400);
  }
}

void VectorDBServer::handleUpdatePersistenceConfig(const httplib::Request& req, httplib::Response& res) {
  total_requests++;
  try {
    auto body = json::parse(req.body);

    PersistenceConfig cfg = db->getPersistenceConfig();

    if (body.contains("checkpoint_interval_ms")) {
      auto ms = body["checkpoint_interval_ms"].get<int64_t>();
      cfg.checkpoint_interval = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::milliseconds(ms));
    }
    if (body.contains("checkpoint_trigger_ops")) cfg.checkpoint_trigger_ops = body["checkpoint_trigger_ops"];
    if (body.contains("log_rotation_size"))      cfg.log_rotation_size      = body["log_rotation_size"];
    if (body.contains("max_log_files"))          cfg.max_log_files          = body["max_log_files"];
    if (body.contains("log_directory"))          cfg.log_directory          = body["log_directory"];
    if (body.contains("data_directory"))         cfg.data_directory         = body["data_directory"];

    std::lock_guard<std::mutex> lock(db_mutex);
    db->updatePersistenceConfig(cfg);

    json r;
    r["success"] = true;
    r["message"] = "Persistence configuration updated successfully";
    res.set_content(r.dump(), "application/json");
    successful_requests++;
    logRequest("PUT", "/config/persistence", 200);
  } catch (const std::exception& e) {
    handleError(res, 400, std::string("Invalid request format: ") + e.what());
    failed_requests++;
    logRequest("PUT", "/config/persistence", 400);
  }
}

void VectorDBServer::logRequest(const std::string& method, const std::string& path, int status_code) {
  auto now = std::chrono::system_clock::now();
  auto ts  = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::cout << "[" << ts << "] " << method << " " << path << " " << status_code << std::endl;
}

void VectorDBServer::handleError(httplib::Response& res, int status_code, const std::string& message) {
  json err;
  err["error"]       = true;
  err["message"]     = message;
  err["status_code"] = status_code;

  res.status = status_code;
  res.set_content(err.dump(), "application/json");
}

void VectorDBServer::handleSuccess(httplib::Response& res, const json& data) {
  json r;
  r["success"] = true;
  if (!data.is_null() && !data.empty()) r["data"] = data;
  res.set_content(r.dump(), "application/json");
}

void VectorDBServer::recoveryMonitorFunction() {
  while (!should_stop_monitoring) {
    try {
      recovery_in_progress = db->isRecovering();
      if (recovery_in_progress) {
        std::cout << "Recovery in progress..." << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::seconds(5));
    } catch (const std::exception& e) {
      std::cerr << "Recovery monitor error: " << e.what() << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  }
}

void VectorDBServer::startRecoveryMonitoring() {
  should_stop_monitoring = false;
  recovery_monitor_thread = std::thread(&VectorDBServer::recoveryMonitorFunction, this);
}

void VectorDBServer::stopRecoveryMonitoring() {
  should_stop_monitoring = true;
  if (recovery_monitor_thread.joinable()) {
    recovery_monitor_thread.join();
  }
}

bool VectorDBServer::validateVector(const json& v, size_t expected_dimensions) {
  if (!v.is_array()) return false;
  if (v.size() != expected_dimensions) return false;
  for (const auto& el : v) {
    if (!el.is_number()) return false;
  }
  return true;
}

bool VectorDBServer::validateBatchRequest(const json& body) {
  if (!body.contains("keys") || !body.contains("vectors")) return false;
  if (!body["keys"].is_array() || !body["vectors"].is_array()) return false;
  if (body["keys"].size() != body["vectors"].size()) return false;
  for (const auto& v : body["vectors"]) {
    if (!validateVector(v, dimensions)) return false;
  }
  return true;
}

void VectorDBServer::start(bool blocking) {
  std::cout << "Starting Vector Database Server on " << host << ":" << port << std::endl;
  db->initialize();
  server.listen(host, port); // blocking either way in this simple server
}

void VectorDBServer::stop() {
  std::cout << "Stopping Vector Database Server..." << std::endl;
  server.stop();
}

bool VectorDBServer::isRunning() const {
  return server.is_running();
}

VectorDBServer::ServerStatistics VectorDBServer::getServerStatistics() const {
  ServerStatistics s;
  s.total_requests        = total_requests.load();
  s.successful_requests   = successful_requests.load();
  s.failed_requests       = failed_requests.load();
  s.total_batch_operations= total_batch_operations.load();
  s.recovery_in_progress  = recovery_in_progress.load();
  s.server_running        = server.is_running();
  s.host                  = host;
  s.port                  = port;
  return s;
}

void VectorDBServer::updateConfig(const std::string& new_host,
                                  int new_port,
                                  bool enable_recovery,
                                  bool enable_batch,
                                  bool enable_stats) {
  host = new_host;
  port = new_port;
  enable_recovery_endpoints   = enable_recovery;
  enable_batch_endpoints      = enable_batch;
  enable_statistics_endpoints = enable_stats;
}
